#!/usr/bin/env python

"""
CmpRuns - A simple tool for comparing two static analyzer runs to determine
which reports have been added, removed, or changed.

This is designed to support automated testing using the static analyzer, from
two perspectives:
  1. To monitor changes in the static analyzer's reports on real code bases,
     for regression testing.

  2. For use by end users who want to integrate regular static analyzer testing
     into a buildbot like environment.

Usage:

    # Load the results of both runs, to obtain lists of the corresponding
    # AnalysisDiagnostic objects.
    #
    resultsA = load_results_from_single_run(singleRunInfoA, delete_empty)
    resultsB = load_results_from_single_run(singleRunInfoB, delete_empty)

    # Generate a relation from diagnostics in run A to diagnostics in run B
    # to obtain a list of triples (a, b, confidence).
    diff = compare_results(resultsA, resultsB)

"""
import argparse
import json
import os
import plistlib
import re
import sys

from math import log
from collections import defaultdict
from copy import copy
from typing import (Any, cast, Dict, List, Optional, Sequence, TextIO, TypeVar,
                    Tuple, Union)


Number = Union[int, float]
Stats = Dict[str, Dict[str, Number]]
Plist = Dict[str, Any]
JSON = Dict[str, Any]
# Type for generics
T = TypeVar('T')

STATS_REGEXP = re.compile(r"Statistics: (\{.+\})", re.MULTILINE | re.DOTALL)


class Colors:
    """
    Color for terminal highlight.
    """
    RED = '\x1b[2;30;41m'
    GREEN = '\x1b[6;30;42m'
    CLEAR = '\x1b[0m'


class SingleRunInfo:
    """
    Information about analysis run:
    path - the analysis output directory
    root - the name of the root directory, which will be disregarded when
    determining the source file name
    """
    def __init__(self, path: str, root: str = "", verbose_log=None):
        self.path = path
        self.root = root.rstrip("/\\")
        self.verbose_log = verbose_log


class AnalysisDiagnostic:
    def __init__(self, data: Plist, report: "AnalysisReport",
                 html_report: Optional[str]):
        self._data = data
        self._loc = self._data['location']
        self._report = report
        self._html_report = html_report
        self._report_size = len(self._data['path'])

    def get_file_name(self) -> str:
        root = self._report.run.root
        file_name = self._report.files[self._loc['file']]

        if file_name.startswith(root) and len(root) > 0:
            return file_name[len(root) + 1:]

        return file_name

    def get_root_file_name(self) -> str:
        path = self._data['path']

        if not path:
            return self.get_file_name()

        p = path[0]
        if 'location' in p:
            file_index = p['location']['file']
        else:  # control edge
            file_index = path[0]['edges'][0]['start'][0]['file']

        out = self._report.files[file_index]
        root = self._report.run.root

        if out.startswith(root):
            return out[len(root):]

        return out

    def get_line(self) -> int:
        return self._loc['line']

    def get_column(self) -> int:
        return self._loc['col']

    def get_path_length(self) -> int:
        return self._report_size

    def get_category(self) -> str:
        return self._data['category']

    def get_description(self) -> str:
        return self._data['description']

    def get_issue_identifier(self) -> str:
        id = self.get_file_name() + "+"

        if "issue_context" in self._data:
            id += self._data["issue_context"] + "+"

        if "issue_hash_content_of_line_in_context" in self._data:
            id += str(self._data["issue_hash_content_of_line_in_context"])

        return id

    def get_html_report(self) -> str:
        if self._html_report is None:
            return " "

        return os.path.join(self._report.run.path, self._html_report)

    def get_readable_name(self) -> str:
        if "issue_context" in self._data:
            funcname_postfix = "#" + self._data["issue_context"]
        else:
            funcname_postfix = ""

        root_filename = self.get_root_file_name()
        file_name = self.get_file_name()

        if root_filename != file_name:
            file_prefix = f"[{root_filename}] {file_name}"
        else:
            file_prefix = root_filename

        line = self.get_line()
        col = self.get_column()
        return f"{file_prefix}{funcname_postfix}:{line}:{col}" \
            f", {self.get_category()}: {self.get_description()}"

    # Note, the data format is not an API and may change from one analyzer
    # version to another.
    def get_raw_data(self) -> Plist:
        return self._data


class AnalysisRun:
    def __init__(self, info: SingleRunInfo):
        self.path = info.path
        self.root = info.root
        self.info = info
        self.reports: List[AnalysisReport] = []
        # Cumulative list of all diagnostics from all the reports.
        self.diagnostics: List[AnalysisDiagnostic] = []
        self.clang_version: Optional[str] = None
        self.raw_stats: List[JSON] = []

    def get_clang_version(self) -> Optional[str]:
        return self.clang_version

    def read_single_file(self, path: str, delete_empty: bool):
        with open(path, "rb") as plist_file:
            data = plistlib.load(plist_file)

        if 'statistics' in data:
            self.raw_stats.append(json.loads(data['statistics']))
            data.pop('statistics')

        # We want to retrieve the clang version even if there are no
        # reports. Assume that all reports were created using the same
        # clang version (this is always true and is more efficient).
        if 'clang_version' in data:
            if self.clang_version is None:
                self.clang_version = data.pop('clang_version')
            else:
                data.pop('clang_version')

        # Ignore/delete empty reports.
        if not data['files']:
            if delete_empty:
                os.remove(path)
            return

        # Extract the HTML reports, if they exists.
        if 'HTMLDiagnostics_files' in data['diagnostics'][0]:
            htmlFiles = []
            for d in data['diagnostics']:
                # FIXME: Why is this named files, when does it have multiple
                # files?
                assert len(d['HTMLDiagnostics_files']) == 1
                htmlFiles.append(d.pop('HTMLDiagnostics_files')[0])
        else:
            htmlFiles = [None] * len(data['diagnostics'])

        report = AnalysisReport(self, data.pop('files'))
        diagnostics = [AnalysisDiagnostic(d, report, h)
                       for d, h in zip(data.pop('diagnostics'), htmlFiles)]

        assert not data

        report.diagnostics.extend(diagnostics)
        self.reports.append(report)
        self.diagnostics.extend(diagnostics)


class AnalysisReport:
    def __init__(self, run: AnalysisRun, files: List[str]):
        self.run = run
        self.files = files
        self.diagnostics: List[AnalysisDiagnostic] = []


def load_results(path: str, args: argparse.Namespace, root: str = "",
                 delete_empty: bool = True) -> AnalysisRun:
    """
    Backwards compatibility API.
    """
    return load_results_from_single_run(SingleRunInfo(path, root,
                                                      args.verbose_log),
                                        delete_empty)


def load_results_from_single_run(info: SingleRunInfo,
                                 delete_empty: bool = True) -> AnalysisRun:
    """
    # Load results of the analyzes from a given output folder.
    # - info is the SingleRunInfo object
    # - delete_empty specifies if the empty plist files should be deleted

    """
    path = info.path
    run = AnalysisRun(info)

    if os.path.isfile(path):
        run.read_single_file(path, delete_empty)
    else:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                if not f.endswith('plist'):
                    continue

                p = os.path.join(dirpath, f)
                run.read_single_file(p, delete_empty)

    return run


def cmp_analysis_diagnostic(d):
    return d.get_issue_identifier()


PresentInBoth = Tuple[AnalysisDiagnostic, AnalysisDiagnostic]
PresentOnlyInOld = Tuple[AnalysisDiagnostic, None]
PresentOnlyInNew = Tuple[None, AnalysisDiagnostic]
ComparisonResult = List[Union[PresentInBoth,
                              PresentOnlyInOld,
                              PresentOnlyInNew]]


def compare_results(results_old: AnalysisRun, results_new: AnalysisRun,
                    args: argparse.Namespace) -> ComparisonResult:
    """
    compare_results - Generate a relation from diagnostics in run A to
    diagnostics in run B.

    The result is the relation as a list of triples (a, b) where
    each element {a,b} is None or a matching element from the respective run
    """

    res: ComparisonResult = []

    # Map size_before -> size_after
    path_difference_data: List[float] = []

    # Quickly eliminate equal elements.
    neq_old: List[AnalysisDiagnostic] = []
    neq_new: List[AnalysisDiagnostic] = []

    diags_old = copy(results_old.diagnostics)
    diags_new = copy(results_new.diagnostics)

    diags_old.sort(key=cmp_analysis_diagnostic)
    diags_new.sort(key=cmp_analysis_diagnostic)

    while diags_old and diags_new:
        a = diags_old.pop()
        b = diags_new.pop()

        if a.get_issue_identifier() == b.get_issue_identifier():
            if a.get_path_length() != b.get_path_length():

                if args.relative_path_histogram:
                    path_difference_data.append(
                        float(a.get_path_length()) / b.get_path_length())

                elif args.relative_log_path_histogram:
                    path_difference_data.append(
                        log(float(a.get_path_length()) / b.get_path_length()))

                elif args.absolute_path_histogram:
                    path_difference_data.append(
                        a.get_path_length() - b.get_path_length())

            res.append((a, b))

        elif a.get_issue_identifier() > b.get_issue_identifier():
            diags_new.append(b)
            neq_old.append(a)

        else:
            diags_old.append(a)
            neq_new.append(b)

    neq_old.extend(diags_old)
    neq_new.extend(diags_new)

    # FIXME: Add fuzzy matching. One simple and possible effective idea would
    # be to bin the diagnostics, print them in a normalized form (based solely
    # on the structure of the diagnostic), compute the diff, then use that as
    # the basis for matching. This has the nice property that we don't depend
    # in any way on the diagnostic format.

    for a in neq_old:
        res.append((a, None))
    for b in neq_new:
        res.append((None, b))

    if args.relative_log_path_histogram or args.relative_path_histogram or \
            args.absolute_path_histogram:
        from matplotlib import pyplot
        pyplot.hist(path_difference_data, bins=100)
        pyplot.show()

    return res


def compute_percentile(values: Sequence[T], percentile: float) -> T:
    """
    Return computed percentile.
    """
    return sorted(values)[int(round(percentile * len(values) + 0.5)) - 1]


def derive_stats(results: AnalysisRun) -> Stats:
    # Assume all keys are the same in each statistics bucket.
    combined_data = defaultdict(list)

    # Collect data on paths length.
    for report in results.reports:
        for diagnostic in report.diagnostics:
            combined_data['PathsLength'].append(diagnostic.get_path_length())

    for stat in results.raw_stats:
        for key, value in stat.items():
            combined_data[str(key)].append(value)

    combined_stats: Stats = {}

    for key, values in combined_data.items():
        combined_stats[key] = {
            "max": max(values),
            "min": min(values),
            "mean": sum(values) / len(values),
            "90th %tile": compute_percentile(values, 0.9),
            "95th %tile": compute_percentile(values, 0.95),
            "median": sorted(values)[len(values) // 2],
            "total": sum(values)
        }

    return combined_stats


# TODO: compare_results decouples comparison from the output, we should
#       do it here as well
def compare_stats(results_old: AnalysisRun, results_new: AnalysisRun):
    stats_old = derive_stats(results_old)
    stats_new = derive_stats(results_new)

    old_keys = set(stats_old.keys())
    new_keys = set(stats_new.keys())
    keys = sorted(old_keys & new_keys)

    for key in keys:
        print(key)

        nested_keys = sorted(set(stats_old[key]) & set(stats_new[key]))

        for nested_key in nested_keys:
            val_old = float(stats_old[key][nested_key])
            val_new = float(stats_new[key][nested_key])

            report = f"{val_old:.3f} -> {val_new:.3f}"

            # Only apply highlighting when writing to TTY and it's not Windows
            if sys.stdout.isatty() and os.name != 'nt':
                if val_new != 0:
                    ratio = (val_new - val_old) / val_new
                    if ratio < -0.2:
                        report = Colors.GREEN + report + Colors.CLEAR
                    elif ratio > 0.2:
                        report = Colors.RED + report + Colors.CLEAR

            print(f"\t {nested_key} {report}")

    removed_keys = old_keys - new_keys
    if removed_keys:
        print(f"REMOVED statistics: {removed_keys}")

    added_keys = new_keys - old_keys
    if added_keys:
        print(f"ADDED statistics: {added_keys}")

    print()


def dump_scan_build_results_diff(dir_old: str, dir_new: str,
                                 args: argparse.Namespace,
                                 delete_empty: bool = True,
                                 out: TextIO = sys.stdout):
    # Load the run results.
    results_old = load_results(dir_old, args, args.root_old, delete_empty)
    results_new = load_results(dir_new, args, args.root_new, delete_empty)

    if args.show_stats:
        compare_stats(results_old, results_new)
    if args.stats_only:
        return

    # Open the verbose log, if given.
    if args.verbose_log:
        auxLog: Optional[TextIO] = open(args.verbose_log, "w")
    else:
        auxLog = None

    diff = compare_results(results_old, results_new, args)
    found_diffs = 0
    total_added = 0
    total_removed = 0

    for res in diff:
        old, new = res
        if old is None:
            # TODO: mypy still doesn't understand that old and new can't be
            #       both Nones, we should introduce a better type solution
            new = cast(AnalysisDiagnostic, new)
            out.write(f"ADDED: {new.get_readable_name()}\n")
            found_diffs += 1
            total_added += 1
            if auxLog:
                auxLog.write(f"('ADDED', {new.get_readable_name()}, "
                             f"{new.get_html_report()})\n")

        elif new is None:
            out.write(f"REMOVED: {old.get_readable_name()}\n")
            found_diffs += 1
            total_removed += 1
            if auxLog:
                auxLog.write(f"('REMOVED', {old.get_readable_name()}, "
                             f"{old.get_html_report()})\n")
        else:
            pass

    total_reports = len(results_new.diagnostics)
    out.write(f"TOTAL REPORTS: {total_reports}\n")
    out.write(f"TOTAL ADDED: {total_added}\n")
    out.write(f"TOTAL REMOVED: {total_removed}\n")

    if auxLog:
        auxLog.write(f"('TOTAL NEW REPORTS', {total_reports})\n")
        auxLog.write(f"('TOTAL DIFFERENCES', {found_diffs})\n")
        auxLog.close()

    # TODO: change to NamedTuple
    return found_diffs, len(results_old.diagnostics), \
        len(results_new.diagnostics)


def generate_option_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-old", dest="root_old",
                        help="Prefix to ignore on source files for "
                        "OLD directory",
                        action="store", type=str, default="")
    parser.add_argument("--root-new", dest="root_new",
                        help="Prefix to ignore on source files for "
                        "NEW directory",
                        action="store", type=str, default="")
    parser.add_argument("--verbose-log", dest="verbose_log",
                        help="Write additional information to LOG "
                        "[default=None]",
                        action="store", type=str, default=None,
                        metavar="LOG")
    parser.add_argument("--relative-path-differences-histogram",
                        action="store_true", dest="relative_path_histogram",
                        default=False,
                        help="Show histogram of relative paths differences. "
                        "Requires matplotlib")
    parser.add_argument("--relative-log-path-differences-histogram",
                        action="store_true",
                        dest="relative_log_path_histogram", default=False,
                        help="Show histogram of log relative paths "
                        "differences. Requires matplotlib")
    parser.add_argument("--absolute-path-differences-histogram",
                        action="store_true", dest="absolute_path_histogram",
                        default=False,
                        help="Show histogram of absolute paths differences. "
                        "Requires matplotlib")
    parser.add_argument("--stats-only", action="store_true", dest="stats_only",
                        default=False, help="Only show statistics on reports")
    parser.add_argument("--show-stats", action="store_true", dest="show_stats",
                        default=False, help="Show change in statistics")
    parser.add_argument("old", nargs=1, help="Directory with old results")
    parser.add_argument("new", nargs=1, help="Directory with new results")

    return parser


def main():
    parser = generate_option_parser()
    args = parser.parse_args()

    dir_old = args.old[0]
    dir_new = args.new[0]

    dump_scan_build_results_diff(dir_old, dir_new, args)


if __name__ == '__main__':
    main()
