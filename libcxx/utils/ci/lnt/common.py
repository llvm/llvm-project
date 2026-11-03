# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple
import argparse
import datetime
import json
import logging
import pathlib
import urllib.error
import urllib.parse
import urllib.request
import uuid


# The User-Agent we send to LNT. This is set explicitly since some deployments
# reject requests made with the default User-Agent of urllib.
USER_AGENT = 'libcxx-lnt-tooling'

# LNT instances can be slow: timeout after that many seconds.
LNT_TIMEOUT_SECONDS = 120

# The format of the timestamps in LNT reports, which are always in UTC.
LNT_TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

# How many results to ask for at once when paging. This is the largest page size LNT allows.
LNT_PAGE_SIZE = 10000

# Commits that cannot change the performance of the library are not worth
# benchmarking. This is the set of paths we consider for libc++ changes.
BENCHMARK_PATHS = ['libcxx/include', 'libcxx/src']


class LNTError(Exception):
    pass


def is_sha(string: str) -> bool:
    return len(string) == 40 and all(c in '0123456789abcdef' for c in string.lower())


def lnt_suite_url(lnt_url: str, test_suite: str) -> str:
    """Return the URL of a test suite in the REST API of a LNT instance."""
    return f'{lnt_url.rstrip("/")}/api/suites/{urllib.parse.quote(test_suite, safe="")}'


def lnt_request(url: str, body: Optional[bytes] = None, token: Optional[str] = None) -> dict:
    """
    Send a request to LNT and return its JSON response. If a body is provided, this issues
    a POST request, otherwise a GET. Errors are raised as urllib raises them.
    """
    headers = {'User-Agent': USER_AGENT}
    if token is not None:
        headers['Authorization'] = f'Bearer {token}'
    if body is not None:
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=LNT_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode())


def lnt_get(url: str) -> Optional[dict]:
    """GET a JSON resource, returning None if the server answers 404."""
    logging.debug(f'GET {url}')
    try:
        return lnt_request(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise LNTError(f'GET {url} failed with HTTP {error.code}: {error.reason}')
    except urllib.error.URLError as error:
        raise LNTError(f'GET {url} failed: {error.reason}')
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LNTError(f'GET {url} returned a malformed response: {error}')


def benchmarked_commits(lnt_url: str, test_suite: str, machine: str,
                        submitted_before: Optional[str] = None) -> Optional[Dict[str, int]]:
    """
    Return {commit: number of runs} for a machine, or None if the instance has no
    such machine. With `submitted_before` (an ISO 8601 timestamp), only the runs
    submitted before that time are counted.
    """
    base = lnt_suite_url(lnt_url, test_suite)
    # Listing runs answers 404 both for an unknown suite and for an unknown machine: tell them apart.
    if lnt_get(base) is None:
        raise LNTError(f'{lnt_url} has no test suite named {test_suite}')

    counts: Dict[str, int] = {}
    params = {'machine': machine, 'limit': LNT_PAGE_SIZE}
    if submitted_before is not None:
        params['submitted_before'] = submitted_before
    while True:
        page = lnt_get(f'{base}/runs?{urllib.parse.urlencode(params)}')
        if page is None:
            if 'cursor' in params:
                raise LNTError(f'machine {machine} disappeared from {lnt_url} while listing its runs')
            return None
        for run in page.get('items', []):
            if 'commit' not in run:
                raise LNTError(f'run {run.get("uuid")} has no commit')
            commit = str(run['commit']).strip().lower() # normalize commit SHAs
            counts[commit] = counts.get(commit, 0) + 1
        cursor = (page.get('cursor') or {}).get('next')
        if cursor is None:
            return counts
        params['cursor'] = cursor


def write_lnt_report(path: pathlib.Path, machine: str, machine_fields: dict, commit: dict, tests: List[dict],
                     measured_at: datetime.datetime, run_parameters: Optional[dict] = None) -> None:
    """Write a LNT v5 run submission, ready to be submitted with `lnt v5 submit`."""
    report = {
        'format_version': '5',
        # Pick the UUID ourselves so that we can safely retry submitting this run if needed without
        # running the risk of storing the same run twice.
        'uuid': str(uuid.uuid4()),
        'measured_at': measured_at.astimezone(datetime.timezone.utc).strftime(LNT_TIMESTAMP_FORMAT),
        'machine': {'name': machine, 'fields': machine_fields},
        'commit': commit,
        'run_parameters': run_parameters or {},
        'tests': tests,
    }
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, allow_nan=False)


def parse_lnt_results(lines: Iterable[str]) -> List[dict]:
    """
    Turn `<test>.<metric> <value>` lines (as produced by consolidate-benchmarks) into the list
    of tests expected in a LNT v5 run submission.

    A test can have several values for the same metric, for example when it was repeated. The
    n-th value of each metric is taken to belong to the n-th sample of that test.
    """
    tests: Dict[str, Dict[str, List[float]]] = {}
    for (number, line) in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            (key, value) = line.split()
            (name, metric) = key.rsplit('.', 1)
            parsed = float(value)
        except ValueError:
            raise ValueError(f'line {number} is not of the form `<test>.<metric> <value>`: {line.strip()!r}')
        tests.setdefault(name, {}).setdefault(metric, []).append(parsed)

    result = []
    for (name, metrics) in tests.items():
        count = max(len(values) for values in metrics.values())
        samples = [{metric: values[i] for (metric, values) in metrics.items() if i < len(values)}
                   for i in range(count)]
        result.append({'name': name, 'samples': samples})
    return result


def at_least(minimum: int) -> Callable[[str], int]:
    """
    Return an argparse type that accepts integers no smaller than `minimum`.

    A value that is not an integer at all is rejected by raising too, since argparse
    otherwise describes that failure using the name of the callable it was handed,
    which is an implementation detail.
    """
    def parse(string: str) -> int:
        try:
            value = int(string)
        except ValueError:
            raise argparse.ArgumentTypeError(f'expected an integer, got {string!r}')
        if value < minimum:
            raise argparse.ArgumentTypeError(f'expected an integer no smaller than {minimum}, got {string}')
        return value
    return parse


def key_value(string: str) -> Tuple[str, str]:
    """An argparse type that accepts KEY=VALUE pairs, where the value may itself contain `=`."""
    if '=' not in string:
        raise argparse.ArgumentTypeError(f'expected KEY=VALUE, got {string!r}')
    (key, value) = string.split('=', 1)
    return (key, value)


class Target(NamedTuple):
    """
    A unit of work: one commit benchmarked on one machine.
    """
    commit: str
    machine: str


class WorkItem(NamedTuple):
    """
    One line of a plan as produced by `plan-benchmarks`.

    A plan is a sequence of work items written one JSON object per line. It is the
    interchange format between `plan-benchmarks` and `dispatch-benchmarks`.
    """
    commit: str
    machine: str
    samples_have: int
    samples_want: int
    reason: str = ''

    @staticmethod
    def parse(line: str) -> 'WorkItem':
        """
        Read one line of a plan, validating every field.

        A plan is normally produced by `plan-benchmarks`, but it can also be
        hand-written, so we are careful about validating the data here.
        """
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f'expected a JSON object, got {type(item).__name__}')

        def field(name: str, types, required: bool = True, default=None):
            if name not in item:
                if required:
                    raise ValueError(f'missing required field {name}')
                return default
            value = item[name]
            if not isinstance(value, types):
                raise ValueError(f'{name} has the wrong type: {value!r}')
            return value

        commit = field('commit', str)
        if not is_sha(commit):
            raise ValueError(f'commit is not a full 40-character SHA: {commit!r}')
        machine = field('machine', str)
        if not machine:
            raise ValueError('machine is empty')
        samples_have = field('samples_have', int)
        samples_want = field('samples_want', int)
        if samples_have < 0:
            raise ValueError(f'samples_have is negative: {samples_have}')
        if samples_want <= samples_have:
            raise ValueError(f'samples_want {samples_want} is not more than samples_have '
                             f'{samples_have}, so there is nothing to request')
        return WorkItem(commit=commit.lower(),
                        machine=machine,
                        samples_have=samples_have,
                        samples_want=samples_want,
                        reason=field('reason', str, required=False, default=''))

    def serialize(self) -> str:
        """Render the item as the single-line JSON object that a plan is made of."""
        return json.dumps(self._asdict(), sort_keys=True)

    @property
    def target(self) -> Target:
        """The unit of work this item is about."""
        return Target(self.commit, self.machine)

    @property
    def runs_to_request(self) -> int:
        """How many more runs this item is asking for."""
        return self.samples_want - self.samples_have
