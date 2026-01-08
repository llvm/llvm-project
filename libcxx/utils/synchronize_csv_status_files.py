#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from typing import List, Dict, Tuple, Optional
import copy
import csv
import itertools
import json
import os
import pathlib
import re
import subprocess

# Number of the 'Libc++ Standards Conformance' project on Github
LIBCXX_CONFORMANCE_PROJECT = '31'

def extract_between_markers(text: str, begin_marker: str, end_marker: str) -> Optional[str]:
    """
    Given a string containing special markers, extract everything located beetwen these markers.

    If the beginning marker is not found, None is returned. If the beginning marker is found but
    there is no end marker, it is an error (this is done to avoid silently accepting inputs that
    are erroneous by mistake).
    """
    start = text.find(begin_marker)
    if start == -1:
        return None

    start += len(begin_marker) # skip the marker itself
    end = text.find(end_marker, start)
    if end == -1:
        raise ArgumentError(f"Could not find end marker {end_marker} in: {text[start:]}")

    return text[start:end]

class PaperStatus:
    TODO = 1
    IN_PROGRESS = 2
    PARTIAL = 3
    DONE = 4
    NOTHING_TO_DO = 5

    _status: int

    _original: Optional[str]
    """
    Optional string from which the paper status was created. This is used to carry additional
    information from CSV rows, like any notes associated to the status.
    """

    def __init__(self, status: int, original: Optional[str] = None):
        self._status = status
        self._original = original

    def __eq__(self, other) -> bool:
        return self._status == other._status

    def __lt__(self, other) -> bool:
        relative_order = {
            PaperStatus.TODO: 0,
            PaperStatus.IN_PROGRESS: 1,
            PaperStatus.PARTIAL: 2,
            PaperStatus.DONE: 3,
            PaperStatus.NOTHING_TO_DO: 3,
        }
        return relative_order[self._status] < relative_order[other._status]

    @staticmethod
    def from_csv_entry(entry: str):
        """
        Parse a paper status out of a CSV row entry. Entries can look like:
        - '' (an empty string, which means the paper is not done yet)
        - '|In Progress|'
        - '|Partial|'
        - '|Complete|'
        - '|Nothing To Do|'
        """
        if entry == '':
            return PaperStatus(PaperStatus.TODO, entry)
        elif entry == '|In Progress|':
            return PaperStatus(PaperStatus.IN_PROGRESS, entry)
        elif entry == '|Partial|':
            return PaperStatus(PaperStatus.PARTIAL, entry)
        elif entry == '|Complete|':
            return PaperStatus(PaperStatus.DONE, entry)
        elif entry == '|Nothing To Do|':
            return PaperStatus(PaperStatus.NOTHING_TO_DO, entry)
        else:
            raise RuntimeError(f'Unexpected CSV entry for status: {entry}')

    @staticmethod
    def from_github_issue(issue: Dict):
        """
        Parse a paper status out of a Github issue obtained from querying a Github project.
        """
        if 'status' not in issue:
            return PaperStatus(PaperStatus.TODO)
        elif issue['status'] == 'Todo':
            return PaperStatus(PaperStatus.TODO)
        elif issue['status'] == 'In Progress':
            return PaperStatus(PaperStatus.IN_PROGRESS)
        elif issue['status'] == 'Partial':
            return PaperStatus(PaperStatus.PARTIAL)
        elif issue['status'] == 'Done':
            return PaperStatus(PaperStatus.DONE)
        elif issue['status'] == 'Nothing To Do':
            return PaperStatus(PaperStatus.NOTHING_TO_DO)
        else:
            raise RuntimeError(f"Received unrecognizable Github issue status: {issue['status']}")

    def to_csv_entry(self) -> str:
        """
        Return the issue state formatted for a CSV entry. The status is formatted as '|Complete|',
        '|In Progress|', etc.
        """
        mapping = {
            PaperStatus.TODO: '',
            PaperStatus.IN_PROGRESS: '|In Progress|',
            PaperStatus.PARTIAL: '|Partial|',
            PaperStatus.DONE: '|Complete|',
            PaperStatus.NOTHING_TO_DO: '|Nothing To Do|',
        }
        return self._original if self._original is not None else mapping[self._status]

class PaperInfo:
    paper_number: str
    """
    Identifier for the paper or the LWG issue. This must be something like 'PnnnnRx', 'Nxxxxx' or 'LWGxxxxx'.
    """

    paper_name: str
    """
    Plain text string representing the name of the paper.
    """

    status: PaperStatus
    """
    Status of the paper/issue. This can be complete, in progress, partial, or done.
    """

    meeting: Optional[str]
    """
    Plain text string representing the meeting at which the paper/issue was voted.
    """

    first_released_version: Optional[str]
    """
    First version of LLVM in which this paper/issue was resolved.
    """

    github_issue: Optional[str]
    """
    Optional number of the Github issue tracking the implementation status of this paper.
    This is used to cross-reference rows in the status pages with Github issues.
    """

    notes: Optional[str]
    """
    Optional plain text string representing notes to associate to the paper.
    This is used to populate the "Notes" column in the CSV status pages.
    """

    original: Optional[object]
    """
    Object from which this PaperInfo originated. This is used to track the CSV row or Github issue that
    was used to generate this PaperInfo and is useful for error reporting purposes.
    """

    def __init__(self, paper_number: str, paper_name: str,
                       status: PaperStatus,
                       meeting: Optional[str] = None,
                       first_released_version: Optional[str] = None,
                       github_issue: Optional[str] = None,
                       notes: Optional[str] = None,
                       original: Optional[object] = None):
        self.paper_number = paper_number
        self.paper_name = paper_name
        self.status = status
        self.meeting = meeting
        self.first_released_version = first_released_version
        self.github_issue = github_issue
        self.notes = notes
        self.original = original

    def for_printing(self) -> Tuple[str, str, str, str, str, str, str]:
        return (
            f'`{self.paper_number} <https://wg21.link/{self.paper_number}>`__',
            self.paper_name,
            self.meeting if self.meeting is not None else '',
            self.status.to_csv_entry(),
            self.first_released_version if self.first_released_version is not None else '',
            f'`#{self.github_issue} <https://github.com/llvm/llvm-project/issues/{self.github_issue}>`__' if self.github_issue is not None else '',
            self.notes if self.notes is not None else '',
        )

    def __repr__(self) -> str:
        return repr(self.original) if self.original is not None else repr(self.for_printing())

    @staticmethod
    def from_csv_row(row: Tuple[str, str, str, str, str, str]):# -> PaperInfo:
        """
        Given a row from one of our status-tracking CSV files, create a PaperInfo object representing that row.
        """
        # Extract the paper number from the first column
        match = re.search(r"((P[0-9R]+)|(LWG[0-9]+)|(N[0-9]+))\s+", row[0])
        if match is None:
            raise RuntimeError(f"Can't parse paper/issue number out of row: {row}")

        # Match the issue number if present
        github_issue = re.search(r'#([0-9]+)', row[5])
        if github_issue:
            github_issue = github_issue.group(1)

        return PaperInfo(
            paper_number=match.group(1),
            paper_name=row[1],
            status=PaperStatus.from_csv_entry(row[3]),
            meeting=row[2] or None,
            first_released_version=row[4] or None,
            github_issue=github_issue,
            notes=row[6] or None,
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

        # Extract any notes from the Github issue and populate the RST notes with them
        issue_description = issue['content']['body']
        notes = extract_between_markers(issue_description, 'BEGIN-RST-NOTES', 'END-RST-NOTES')
        notes = notes.strip() if notes is not None else notes

        return PaperInfo(
            paper_number=paper,
            paper_name=issue['title'].removeprefix(paper + ': '),
            status=PaperStatus.from_github_issue(issue),
            meeting=issue.get('meeting Voted', None),
            first_released_version=None, # TODO
            github_issue=str(issue['content']['number']),
            notes=notes,
            original=issue,
        )

def merge(paper: PaperInfo, gh: PaperInfo) -> PaperInfo:
    """
    Merge a paper coming from a CSV row with a corresponding Github-tracked paper.

    If the CSV row has a status that is "less advanced" than the Github issue, simply update the CSV
    row with the newer status. Otherwise, report an error if they have a different status because
    something must be wrong.

    We don't update issues from 'To Do' to 'In Progress', since that only creates churn and the
    status files aim to document user-facing functionality in releases, for which 'In Progress'
    is not useful.

    In case we don't update the CSV row's status, we still take any updated notes coming
    from the Github issue and we add a link to the Github issue if it was previously missing.
    """
    took_gh_in_full = False # Whether we updated the entire PaperInfo from the Github version
    if paper.status == PaperStatus(PaperStatus.TODO) and gh.status == PaperStatus(PaperStatus.IN_PROGRESS):
        result = copy.deepcopy(paper)
    elif paper.status < gh.status:
        result = copy.deepcopy(gh)
        took_gh_in_full = True
    elif paper.status == gh.status:
        result = copy.deepcopy(paper)
    else:
        print(f"We found a CSV row and a Github issue with different statuses:\nrow: {paper}\nGithub issue: {gh}")
        result = copy.deepcopy(paper)

    # If we didn't take the Github issue in full, make sure to update the notes, the link and anything else.
    if not took_gh_in_full:
        result.github_issue = gh.github_issue
        result.notes = gh.notes
    return result

def load_csv(file: pathlib.Path) -> List[Tuple]:
    rows = []
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    return rows

def write_csv(output: pathlib.Path, rows: List[Tuple]):
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator='\n')
        for row in rows:
            writer.writerow(row)

def create_github_issue(paper: PaperInfo, labels: List[str]) -> None:
    """
    Create a new Github issue representing the given PaperInfo.
    """
    assert paper.github_issue is None, "Trying to create a Github issue for a paper that is already tracked"

    paper_name = paper.paper_name.replace('``', '`').replace('\\', '')

    create_cli = ['gh', 'issue', 'create', '--repo', 'llvm/llvm-project',
                    '--title', f'{paper.paper_number}: {paper_name}',
                    '--body', f'**Link:** https://wg21.link/{paper.paper_number}',
                    '--project', 'libc++ Standards Conformance',
                    '--label', 'libc++']

    for label in labels:
        create_cli += ['--label', label]

    print("Do you want to create the following issue?")
    print(create_cli)
    answer = input("y/n: ")
    if answer == 'n':
        print("Not creating issue")
        return
    elif answer != 'y':
        print(f"Invalid answer {answer}, skipping")
        return

    print("Creating issue")
    issue_link = subprocess.check_output(create_cli).decode().strip()
    print(f"Created tracking issue for {paper.paper_number}: {issue_link}")

    # Retrieve the "Github project item ID" by re-adding the issue to the project again,
    # even though we created it inside the project in the first place.
    item_add_cli = ['gh', 'project', 'item-add', LIBCXX_CONFORMANCE_PROJECT, '--owner', 'llvm', '--url', issue_link, '--format', 'json']
    item = json.loads(subprocess.check_output(item_add_cli).decode().strip())

    # Then, adjust the 'Meeting Voted' field of that item.
    meeting_voted_cli = ['gh', 'project', 'item-edit',
                                '--project-id', 'PVT_kwDOAQWwKc4AlOgt',
                                '--field-id', 'PVTF_lADOAQWwKc4AlOgtzgdUEXI', '--text', paper.meeting,
                                '--id', item['id']]
    subprocess.check_call(meeting_voted_cli)

    # And also adjust the 'Status' field of the item to 'To Do'.
    status_cli = ['gh', 'project', 'item-edit',
                                '--project-id', 'PVT_kwDOAQWwKc4AlOgt',
                                '--field-id', 'PVTSSF_lADOAQWwKc4AlOgtzgdUBak', '--single-select-option-id', 'f75ad846',
                                '--id', item['id']]
    subprocess.check_call(status_cli)

def sync_csv(rows: List[Tuple], from_github: List[PaperInfo], create_new: bool, labels: List[str] = None) -> List[Tuple]:
    """
    Given a list of CSV rows representing an existing status file and a list of PaperInfos representing
    up-to-date (but potentially incomplete) tracking information from Github, this function returns the
    new CSV rows synchronized with the up-to-date information.

    If `create_new` is True and a paper from the CSV file is not tracked on Github yet, this also prompts
    to create a new issue on Github for tracking it. In that case the created issue is tagged with the
    provided labels.

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

        # Find any Github issues tracking this paper. Each row must have one and exactly one Github
        # issue tracking it, which we validate below.
        tracking = [gh for gh in from_github if paper.paper_number == gh.paper_number]

        # If there's more than one tracking issue, something is weird.
        if len(tracking) > 1:
            print(f"Found a row with more than one tracking issue: {row}\ntracked by: {tracking}")
            results.append(row)
            continue

        # Validate the Github issue associated to the CSV row, if any
        if paper.github_issue is not None:
            if len(tracking) == 0:
                print(f"Found row claiming to have a tracking issue, but failed to find a tracking issue on Github: {row}")
                results.append(row)
                continue
            if len(tracking) == 1 and paper.github_issue != tracking[0].github_issue:
                print(f"Found row with incorrect tracking issue: {row}\ntracked by: {tracking[0]}")
                results.append(row)
                continue

        # If there is no tracking issue for that row and we are creating new issues, do that.
        # Otherwise just log that we're missing an issue.
        if len(tracking) == 0:
            if create_new:
                assert labels is not None, "Missing labels when creating new Github issues"
                create_github_issue(paper, labels=labels)
            else:
                print(f"Can't find any Github issue for CSV row: {row}")
            results.append(row)
            continue

        results.append(merge(paper, tracking[0]).for_printing())

    return results

CSV_FILES_TO_SYNC = {
    'Cxx17Issues.csv': ['c++17', 'lwg-issue'],
    'Cxx17Papers.csv': ['c++17', 'wg21 paper'],
    'Cxx20Issues.csv': ['c++20', 'lwg-issue'],
    'Cxx20Papers.csv': ['c++20', 'wg21 paper'],
    'Cxx23Issues.csv': ['c++23', 'lwg-issue'],
    'Cxx23Papers.csv': ['c++23', 'wg21 paper'],
    'Cxx2cIssues.csv': ['c++26', 'lwg-issue'],
    'Cxx2cPapers.csv': ['c++26', 'wg21 paper'],
}

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(prog='synchronize-status-files',
        description='Synchronize the libc++ conformance status files with Github issues')
    parser.add_argument('--validate-only', action='store_true',
        help="Only perform the data validation of CSV files.")
    parser.add_argument('--create-new', action='store_true',
        help="Create new Github issues for CSV rows that do not correspond to any existing Github issue.")
    parser.add_argument('--load-github-from', type=str,
        help="A json file to load the Github project information from instead of querying the API. This is useful for testing to avoid rate limiting.")
    args = parser.parse_args(argv)

    libcxx_root = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Perform data validation for all the CSV files.
    print("Performing data validation of the CSV files")
    for filename in CSV_FILES_TO_SYNC:
        csv = load_csv(libcxx_root / 'docs' / 'Status' / filename)
        for row in csv[1:]: # Skip the header
            if row[0] != "": # Skip separator rows
                PaperInfo.from_csv_row(row)

    if args.validate_only:
        return

    # Load all the Github issues tracking papers from Github.
    if args.load_github_from:
        print(f"Loading all issues from {args.load_github_from}")
        with open(args.load_github_from, 'r', encoding='utf-8') as f:
            project_info = json.load(f)
    else:
        print("Loading all issues from Github")
        gh_command_line = ['gh', 'project', 'item-list', LIBCXX_CONFORMANCE_PROJECT, '--owner', 'llvm', '--format', 'json', '--limit', '9999999']
        project_info = json.loads(subprocess.check_output(gh_command_line))
    from_github = [PaperInfo.from_github_issue(i) for i in project_info['items']]

    # Synchronize CSV files with the Github issues.
    for (filename, labels) in CSV_FILES_TO_SYNC.items():
        print(f"Synchronizing {filename} with Github issues")
        file = libcxx_root / 'docs' / 'Status' / filename
        csv = load_csv(file)
        synced = sync_csv(csv, from_github, create_new=args.create_new, labels=labels)
        write_csv(file, synced)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
