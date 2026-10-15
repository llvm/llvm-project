# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import shutil
import re
import subprocess

from . import (
    build_graph,
    config,
    config_orchestrator,
    failure_logs,
    interposition,
    logging_diagnostics,
    options,
)
from .ssaf_artifact_repository import (
    SSAFArtifactKind,
    SSAFArtifactRepository,
    get_ssaf_artifact_id,
)


def warn_if_not_make(build_command):
    supported = frozenset(["make", "gmake"])
    for word in build_command:
        if word in ("env", "/usr/bin/env") or re.match(r"^[A-Za-z_]\w*=", word):
            continue
        if os.path.basename(word) in supported:
            return
        not_make = word
        break
    else:
        return
    logging_diagnostics.report_warning(
        "'%s' is not make; -B will be appended to it anyway, and "
        "nothing forces the rebuild the artifacts need" % not_make
    )


def write_pass_config(workflow_config, pass_config):
    path = os.path.join(workflow_config.output_dir, pass_config.file_name)
    document = workflow_config.to_json()
    document.update(pass_config.to_json())
    temporary = "%s.%d.tmp" % (path, os.getpid())
    with open(temporary, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return path


def _exit_status(returncode):
    return 128 - returncode if returncode < 0 else returncode


def run_build_with_toolchain_interposition(
    workflow_config, serialized_config_path, shim_link_dir_path, failure_description
):
    # Without -B, make skips an up-to-date artifact, so it is never analysed.
    argv = workflow_config.build_command + ["-B"]
    logging_diagnostics.report_note(workflow_config.verbose, " ".join(argv))
    try:
        status = _exit_status(
            subprocess.call(
                argv,
                env=interposition.make_environment(
                    serialized_config_path,
                    shim_link_dir_path,
                    workflow_config.build_environment,
                ),
            )
        )
    except OSError as error:
        return options.Problem(127, "cannot run '%s': %s" % (argv[0], error))
    if status != 0:
        return options.Problem(
            status, "%s failed (exit %d)" % (failure_description, status)
        )
    return None


def clear_output_dir(output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    open(config.output_dir_marker_path(output_dir), "w").close()


def _no_link_unit(message):
    return None, None, None, options.Problem(1, message)


def get_topmost_link_unit(ssaf_artifact_repo, output_dir):
    graph = build_graph.BuildGraph(config.build_graph_dir(output_dir))
    candidates = graph.find_topmost_outputs(graph.get_recorded_build_steps())
    if not candidates:
        return _no_link_unit(
            "the build produced no link unit; whole-program analysis needs one, "
            "so build a target that links"
        )
    if len(candidates) > 1:
        return _no_link_unit(
            "the build produced %d unconsumed outputs and none of them consumes "
            "the others: %s. Build one of them on its own."
            % (len(candidates), ", ".join(candidates))
        )

    build_artifact_id = build_graph.BuildArtifactId(candidates[0])
    ssaf_artifact_id = get_ssaf_artifact_id(build_artifact_id, SSAFArtifactKind.SUMMARY)
    if not ssaf_artifact_repo.has_artifact(ssaf_artifact_id):
        return _no_link_unit(
            "no summary for '%s'; the SSAF step that produces it did not run "
            "or failed" % build_artifact_id.get_path()
        )
    summary = ssaf_artifact_repo.get_path_for_artifact(ssaf_artifact_id)
    unit_id = ssaf_artifact_repo.get_link_unit_name(summary)
    if unit_id is None:
        return _no_link_unit(
            "'%s' is not a link unit; whole-program analysis needs one, so "
            "build a target that links" % build_artifact_id.get_path()
        )
    return build_artifact_id, summary, unit_id, None


def run_ssaf_analysis_pass(
    workflow_config, analysis_config, ssaf_artifact_repo, shim_link_dir_path
):
    serialized_config_path = write_pass_config(workflow_config, analysis_config)
    problem = run_build_with_toolchain_interposition(
        workflow_config, serialized_config_path, shim_link_dir_path, "build"
    )
    if problem is not None:
        return problem

    summaries = ssaf_artifact_repo.count_summaries()
    if summaries == 0:
        return options.Problem(1, "no summaries were extracted")

    link_unit, summary, _, problem = get_topmost_link_unit(
        ssaf_artifact_repo, workflow_config.output_dir
    )
    if problem is not None:
        return problem

    logging_diagnostics.report_summaries(summaries, link_unit.get_path(), summary)

    if analysis_config.analyses:
        analyze = [
            workflow_config.ssaf_analyzer,
            summary,
            "-o",
            config.wpa_result_path(workflow_config.output_dir),
        ]
        for name in analysis_config.analyses:
            analyze += ["-a", name]
        logging_diagnostics.report_note(workflow_config.verbose, " ".join(analyze))
        status = _exit_status(subprocess.call(analyze))
        if status != 0:
            return options.Problem(
                status, "whole-program analysis failed (exit %d)" % status
            )
        logging_diagnostics.report_analysis_result(
            config.wpa_result_path(workflow_config.output_dir)
        )

    return None


def run_ssaf_transformation_pass(
    workflow_config, transformation_config, ssaf_artifact_repo, shim_link_dir_path
):
    link_unit, _, unit_id, problem = get_topmost_link_unit(
        ssaf_artifact_repo, workflow_config.output_dir
    )
    if problem is not None:
        return problem

    transformation_config.name_link_unit(unit_id)
    serialized_config_path = write_pass_config(workflow_config, transformation_config)
    problem = run_build_with_toolchain_interposition(
        workflow_config,
        serialized_config_path,
        shim_link_dir_path,
        "rebuild for the transformation pass",
    )
    if problem is not None:
        return problem

    edits = get_ssaf_artifact_id(link_unit, SSAFArtifactKind.EDITS)
    if not ssaf_artifact_repo.has_artifact(edits):
        return options.Problem(
            1, "no merged source edits for '%s'" % link_unit.get_path()
        )
    logging_diagnostics.report_transformation(
        ssaf_artifact_repo.get_path_for_artifact(edits),
        ssaf_artifact_repo.get_conflicts(),
    )
    return None


def run_ssaf_workflow(workflow_config, analysis_config, transformation_config):
    warn_if_not_make(workflow_config.build_command)

    ssaf_artifact_repo = SSAFArtifactRepository(
        config.ssaf_artifact_dir(workflow_config.output_dir)
    )
    try:
        clear_output_dir(workflow_config.output_dir)
        shim_link_dir_path = interposition.create_shim_links(
            config.shim_link_dir(workflow_config.output_dir),
            workflow_config.installed_shim_dir_path,
        )
    except OSError as error:
        return options.Problem(
            options.ENVIRONMENT_ERROR,
            "cannot prepare the output directory: %s" % error,
        )

    problem = run_ssaf_analysis_pass(
        workflow_config, analysis_config, ssaf_artifact_repo, shim_link_dir_path
    )
    if problem is not None:
        return problem

    if transformation_config is not None:
        problem = run_ssaf_transformation_pass(
            workflow_config,
            transformation_config,
            ssaf_artifact_repo,
            shim_link_dir_path,
        )
        if problem is not None:
            return problem

    logging_diagnostics.report_output_dir(workflow_config.output_dir)

    failures = failure_logs.count_failure_logs(workflow_config.output_dir)
    if failures:
        return options.Problem(
            0 if workflow_config.keep_going else 1,
            "%d SSAF step(s) failed; see the log files under %s"
            % (failures, workflow_config.output_dir),
        )
    return None


def main(argv=None, env=None):
    env = dict(os.environ) if env is None else env
    command_line_opts, problem = options.resolve(argv)
    if problem is None:
        configs, problem = config_orchestrator.configure(command_line_opts, env)
    if problem is None:
        problem = run_ssaf_workflow(*configs)
    if problem is not None:
        logging_diagnostics.report_error(problem.message, problem.details)
        return problem.exit_status
    return 0
