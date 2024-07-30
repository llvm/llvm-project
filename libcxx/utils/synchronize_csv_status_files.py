#!/usr/bin/env python3

from typing import List, Dict, Tuple, Optional
import csv
import itertools
import json
import os
import pathlib
import re
import subprocess

# Number of the 'Libc++ Standards Conformance' project on Github
LIBCXX_CONFORMANCE_PROJECT = '31'

class PaperInfo:
    paper_number: str
    """
    Identifier for the paper or the LWG issue. This must be something like 'PnnnnRx', 'Nxxxxx' or 'LWGxxxxx'.
    """

    paper_name: str
    """
    Plain text string representing the name of the paper.
    """

    meeting: Optional[str]
    """
    Plain text string representing the meeting at which the paper/issue was voted.
    """

    status: Optional[str]
    """
    Status of the paper/issue. This must be '|Complete|', '|Nothing To Do|', '|In Progress|',
    '|Partial|' or 'Resolved by <something>'.
    """

    first_released_version: Optional[str]
    """
    First version of LLVM in which this paper/issue was resolved.
    """

    labels: Optional[List[str]]
    """
    List of labels to associate to the issue in the status-tracking table. Supported labels are
    'format', 'ranges', 'spaceship', 'flat_containers', 'concurrency TS' and 'DR'.
    """

    original: Optional[object]
    """
    Object from which this PaperInfo originated. This is used to track the CSV row or Github issue that
    was used to generate this PaperInfo and is useful for error reporting purposes.
    """

    def __init__(self, paper_number: str, paper_name: str,
                       meeting: Optional[str] = None,
                       status: Optional[str] = None,
                       first_released_version: Optional[str] = None,
                       labels: Optional[List[str]] = None,
                       original: Optional[object] = None):
        self.paper_number = paper_number
        self.paper_name = paper_name
        self.meeting = meeting
        self.status = status
        self.first_released_version = first_released_version
        self.labels = labels
        self.original = original

    def for_printing(self) -> Tuple[str, str, str, str, str, str]:
        return (
            f'`{self.paper_number} <https://wg21.link/{self.paper_number}>`__',
            self.paper_name,
            self.meeting if self.meeting is not None else '',
            self.status if self.status is not None else '',
            self.first_released_version if self.first_released_version is not None else '',
            ' '.join(f'|{label}|' for label in self.labels) if self.labels is not None else '',
        )

    def __repr__(self) -> str:
        return repr(self.original) if self.original is not None else repr(self.for_printing())

    def is_implemented(self) -> bool:
        if self.status is None:
            return False
        if re.search(r'(in progress|partial)', self.status.lower()):
            return False
        return True

    @staticmethod
    def from_csv_row(row: Tuple[str, str, str, str, str, str]):# -> PaperInfo:
        """
        Given a row from one of our status-tracking CSV files, create a PaperInfo object representing that row.
        """
        # Extract the paper number from the first column
        match = re.search(r"((P[0-9R]+)|(LWG[0-9]+)|(N[0-9]+))\s+", row[0])
        if match is None:
            raise RuntimeError(f"Can't parse paper/issue number out of row: {row}")

        return PaperInfo(
            paper_number=match.group(1),
            paper_name=row[1],
            meeting=row[2] or None,
            status=row[3] or None,
            first_released_version=row[4] or None,
            labels=[l.strip('|') for l in row[5].split(' ') if l] or None,
            original=row,
        )

    @staticmethod
    def from_github_issue(issue: Dict):# -> PaperInfo:
        """
        Create a PaperInfo object from the Github issue information obtained from querying a Github Project.
        """
        # Extract the paper number from the issue title
        match = re.search(r"((P[0-9R]+)|(LWG[0-9]+)|(N[0-9]+)):", issue['title'])
        if match is None:
            raise RuntimeError(f"Issue doesn't have a title that we know how to parse: {issue}")
        paper = match.group(1)

        # Figure out the status of the paper according to the Github project information.
        #
        # Sadly, we can't make a finer-grained distiction about *how* the issue
        # was closed (such as Nothing To Do or similar).
        status = '|Complete|' if 'status' in issue and issue['status'] == 'Done' else None

        # Handle labels
        valid_labels = ('format', 'ranges', 'spaceship', 'flat_containers', 'concurrency TS', 'DR')
        labels = [label for label in issue['labels'] if label in valid_labels]

        return PaperInfo(
            paper_number=paper,
            paper_name=issue['title'],
            meeting=issue.get('meeting Voted', None),
            status=status,
            first_released_version=None, # TODO
            labels=labels if labels else None,
            original=issue,
        )

def load_csv(file: pathlib.Path) -> List[Tuple]:
    rows = []
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    return rows

def write_csv(output: pathlib.Path, rows: List[Tuple]):
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator='\n')
        for row in rows:
            writer.writerow(row)

def sync_csv(rows: List[Tuple], from_github: List[PaperInfo]) -> List[Tuple]:
    """
    Given a list of CSV rows representing an existing status file and a list of PaperInfos representing
    up-to-date (but potentially incomplete) tracking information from Github, this function returns the
    new CSV rows synchronized with the up-to-date information.

    Note that this only tracks changes from 'not implemented' issues to 'implemented'. If an up-to-date
    PaperInfo reports that a paper is not implemented but the existing CSV rows report it as implemented,
    it is an error (i.e. the result is not a CSV row where the paper is *not* implemented).
    """
    results = [rows[0]] # Start with the header
    for row in rows[1:]: # Skip the header
        # If the row contains empty entries, this is a "separator row" between meetings.
        # Preserve it as-is.
        if row[0] == "":
            results.append(row)
            continue

        paper = PaperInfo.from_csv_row(row)

        # If the row is already implemented, basically keep it unchanged but also validate that we're not
        # out-of-sync with any still-open Github issue tracking the same paper.
        if paper.is_implemented():
            dangling = [gh for gh in from_github if gh.paper_number == paper.paper_number and not gh.is_implemented()]
            if dangling:
                raise RuntimeError(f"We found the following open tracking issues for a row which is already marked as implemented:\nrow: {row}\ntracking issues: {dangling}")
            results.append(paper.for_printing())
        else:
            # Find any Github issues tracking this paper
            tracking = [gh for gh in from_github if paper.paper_number == gh.paper_number]

            # If there is no tracking issue for that row in the CSV, this is an error since we're
            # missing a Github issue.
            if not tracking:
                raise RuntimeError(f"Can't find any Github issue for CSV row which isn't marked as done yet: {row}")

            # If there's more than one tracking issue, something is weird too.
            if len(tracking) > 1:
                raise RuntimeError(f"Found a row with more than one tracking issue: {row}\ntracked by: {tracking}")

            # If the issue is closed, synchronize the row based on the Github issue. Otherwise, use the
            # existing CSV row as-is.
            results.append(tracking[0].for_printing() if tracking[0].is_implemented() else row)

    return results

CSV_FILES_TO_SYNC = [
    'Cxx14Issues.csv',
    'Cxx17Issues.csv',
    'Cxx17Papers.csv',
    'Cxx20Issues.csv',
    'Cxx20Papers.csv',
    # TODO: The Github issues are not created yet.
    # 'Cxx23Issues.csv',
    # 'Cxx23Papers.csv',
    # 'Cxx2cIssues.csv',
    # 'Cxx2cPapers.csv',
]

def main():
    libcxx_root = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Extract the list of PaperInfos from issues we're tracking on Github.
    print("Loading all issues from Github")
    gh_command_line = ['gh', 'project', 'item-list', LIBCXX_CONFORMANCE_PROJECT, '--owner', 'llvm', '--format', 'json', '--limit', '9999999']
    project_info = json.loads(subprocess.check_output(gh_command_line))
    from_github = [PaperInfo.from_github_issue(i) for i in project_info['items']]

    for filename in CSV_FILES_TO_SYNC:
        print(f"Synchronizing {filename} with Github issues")
        file = libcxx_root / 'docs' / 'Status' / filename
        csv = load_csv(file)
        synced = sync_csv(csv, from_github)
        write_csv(file, synced)

if __name__ == '__main__':
    main()
