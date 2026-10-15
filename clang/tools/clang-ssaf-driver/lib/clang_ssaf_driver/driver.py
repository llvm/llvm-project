# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from . import (
    logging_diagnostics,
    make_orchestrator,
    options,
    patch,
    report,
    source_edits,
)
from .transformations import TRANSFORMATIONS


def run(command_line_opts):
    orchestrator = make_orchestrator.find_orchestrator()
    if not os.path.isfile(orchestrator):
        return options.Problem(
            options.ENVIRONMENT_ERROR,
            "cannot find %s: %s" % (make_orchestrator.ORCHESTRATOR, orchestrator),
        )
    project_root = os.path.realpath(os.getcwd())
    output_dir = os.path.realpath(command_line_opts.output_dir)
    status, results = make_orchestrator.run_orchestrator(
        make_orchestrator.orchestrator_command(
            orchestrator,
            TRANSFORMATIONS[command_line_opts.transformation],
            command_line_opts,
        ),
        command_line_opts.verbose,
    )
    if status != 0:
        return options.Problem(
            status, "%s failed (exit %d)" % (make_orchestrator.ORCHESTRATOR, status)
        )
    if "source-edits" not in results:
        return options.Problem(
            options.ENVIRONMENT_ERROR,
            "%s reported no source edits" % make_orchestrator.ORCHESTRATOR,
        )

    replacements, problem = source_edits.read_source_edits(results["source-edits"])
    if problem is not None:
        return problem
    log, problem = report.load(
        [results.get("transformation-report"), results.get("conflict-report")]
    )
    if problem is not None:
        return problem
    collected, problem = patch.collect(
        replacements, project_root, report.conflict_ranges(log)
    )
    if problem is not None:
        return problem

    for line in report.diagnostics(log, project_root):
        logging_diagnostics.report_finding(line)
    for path in collected.outside:
        logging_diagnostics.report_warning(
            "edits to a file outside the project were left out: %s" % path
        )
    for relative_path, count in collected.blocked:
        logging_diagnostics.report_warning(
            "left out %d edit(s) to %s that overlap a reported conflict"
            % (count, relative_path)
        )

    patch_path = os.path.join(output_dir, command_line_opts.transformation + ".patch")
    report_path = os.path.join(output_dir, command_line_opts.transformation + ".sarif")
    try:
        with open(patch_path, "wb") as handle:
            handle.write(patch.render(collected.files))
        report.write(report_path, log)
    except OSError as error:
        return options.Problem(
            options.ENVIRONMENT_ERROR, "cannot write the results: %s" % error
        )
    logging_diagnostics.report_info("patch", patch_path)
    logging_diagnostics.report_info("report", report_path)
    return None


def main(argv=None):
    command_line_opts, problem = options.resolve(argv)
    if problem is None:
        problem = run(command_line_opts)
    if problem is not None:
        logging_diagnostics.report_error(problem.message, problem.details)
        return problem.exit_status
    return 0
