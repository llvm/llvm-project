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
                       notes: Optional[str] = None,
                       original: Optional[object] = None):
        self.paper_number = paper_number
        self.paper_name = paper_name
        self.status = status
        self.meeting = meeting
        self.first_released_version = first_released_version
        self.notes = notes
        self.original = original

    def for_printing(self) -> Tuple[str, str, str, str, str, str]:
        return (
            f'`{self.paper_number} <https://wg21.link/{self.paper_number}>`__',
            self.paper_name,
            self.meeting if self.meeting is not None else '',
            self.status.to_csv_entry(),
            self.first_released_version if self.first_released_version is not None else '',
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

        return PaperInfo(
            paper_number=match.group(1),
            paper_name=row[1],
            status=PaperStatus.from_csv_entry(row[3]),
            meeting=row[2] or None,
            first_released_version=row[4] or None,
            notes=row[5] or None,
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
            paper_name=issue['title'],
            status=PaperStatus.from_github_issue(issue),
            meeting=issue.get('meeting Voted', None),
            first_released_version=None, # TODO
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
    from the Github issue.
    """
    if paper.status == PaperStatus(PaperStatus.TODO) and gh.status == PaperStatus(PaperStatus.IN_PROGRESS):
        result = copy.deepcopy(paper)
        result.notes = gh.notes
    elif paper.status < gh.status:
        result = copy.deepcopy(gh)
    elif paper.status == gh.status:
        result = copy.deepcopy(paper)
        result.notes = gh.notes
    else:
        print(f"We found a CSV row and a Github issue with different statuses:\nrow: {paper}\nGithub issue: {gh}")
        result = copy.deepcopy(paper)
    return result

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

        # Find any Github issues tracking this paper. Each row must have one and exactly one Github
        # issue tracking it, which we validate below.
        tracking = [gh for gh in from_github if paper.paper_number == gh.paper_number]

        # If there is no tracking issue for that row in the CSV, this is an error since we're
        # missing a Github issue.
        if len(tracking) == 0:
            print(f"Can't find any Github issue for CSV row: {row}")
            results.append(row)
            continue

        # If there's more than one tracking issue, something is weird too.
        if len(tracking) > 1:
            print(f"Found a row with more than one tracking issue: {row}\ntracked by: {tracking}")
            results.append(row)
            continue

        results.append(merge(paper, tracking[0]).for_printing())

    return results

CSV_FILES_TO_SYNC = [
    'Cxx17Issues.csv',
    'Cxx17Papers.csv',
    'Cxx20Issues.csv',
    'Cxx20Papers.csv',
    'Cxx23Issues.csv',
    'Cxx23Papers.csv',
    'Cxx2cIssues.csv',
    'Cxx2cPapers.csv',
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
