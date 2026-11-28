# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from . import PROGRAM


def report_note(verbose, message):
    if verbose:
        sys.stderr.write("%s: note: %s\n" % (PROGRAM, message))


def report_warning(message):
    sys.stderr.write("%s: warning: %s\n" % (PROGRAM, message))


def report_error(message, details=()):
    sys.stderr.write("%s: error: %s\n" % (PROGRAM, message))
    for detail in details:
        sys.stderr.write("  %s\n" % detail)


def report_info(key, value):
    sys.stdout.write("%s: %s\n" % (key, value))
    sys.stdout.flush()


def report_output_dir(output_dir):
    report_info("output-dir", output_dir)


def report_summaries(summaries, link_unit, summary_path):
    report_info("summaries", summaries)
    report_info("link-unit", link_unit)
    report_info("link-unit-summary", summary_path)


def report_analysis_result(wpa_result_path):
    report_info("analysis-result", wpa_result_path)


def report_transformation(merged, transformation_report, conflict_report):
    report_info("source-edits", merged)
    if transformation_report:
        report_info("transformation-report", transformation_report)
    if conflict_report:
        report_info("conflict-report", conflict_report)
