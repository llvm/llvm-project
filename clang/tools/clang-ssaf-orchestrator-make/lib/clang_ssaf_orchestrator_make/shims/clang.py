# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import subprocess

from .. import config, logging_diagnostics, utils
from ..build_graph import BuildArtifactId
from ..failure_logs import SSAFFailureKind
from ..ssaf_artifact_repository import SSAFArtifactKind, get_ssaf_artifact_id
from . import run
from .driver_jobs import Job, parse_driver_job_listing
from .shim import SKIP_BINARY_TOOL_INVOCATION


def _shim_name_for(tool):
    """Strip the directory and any version suffix from a tool's path."""
    return re.sub(r"-\d+$", "", os.path.basename(tool))


def _output_build_artifact_id(job, cwd):
    if not job.output:
        return None
    return BuildArtifactId(os.path.join(cwd, job.output))


def _invoke_shim_for_job(shim, job, cwd):
    name = _shim_name_for(job.argv[0]) if job.argv else ""
    link = os.path.join(shim.shim_link_dir, name) if name else ""
    if not link or not os.path.isfile(link):
        logging_diagnostics.report_note(
            shim.workflow_config.verbose,
            "no shim stands in for %s; the compiler has already run it" % name,
        )
        return
    shim.run(
        [link] + job.argv[1:] + [SKIP_BINARY_TOOL_INVOCATION],
        SSAFFailureKind.FORWARD,
        _output_build_artifact_id(job, cwd),
        relay_output=True,
    )


def _get_driver_jobs(shim, listing_command, cwd):
    logging_diagnostics.report_note(
        shim.workflow_config.verbose, "binary: %s" % " ".join(listing_command)
    )
    completed = subprocess.run(
        listing_command,
        cwd=cwd,
        env=shim.get_unmodified_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return parse_driver_job_listing(completed.stderr), completed


def _invoke_all_jobs(shim, bin_tool_path, cwd):
    listing_command = [bin_tool_path, "-###"] + shim.argv[1:]
    jobs, completed = _get_driver_jobs(shim, listing_command, cwd)
    if not jobs:
        if completed.returncode != 0:
            shim.log_failure(
                listing_command,
                completed,
                SSAFFailureKind.DRIVER_JOBS,
                _output_build_artifact_id(Job(list(shim.argv[1:])), cwd),
            )
        else:
            logging_diagnostics.report_note(
                shim.workflow_config.verbose, "no jobs for %s" % " ".join(shim.argv[1:])
            )
        return

    for job in jobs:
        _invoke_shim_for_job(shim, job, cwd)


def _summary_extraction_args(shim, build_artifact_id):
    ssaf_artifact_repo = shim.ssaf_artifact_repo
    summary = utils.make_parents(
        ssaf_artifact_repo.get_path_for_artifact(
            get_ssaf_artifact_id(build_artifact_id, SSAFArtifactKind.SUMMARY)
        )
    )
    extra = [
        "--ssaf-extract-summaries=" + shim.pass_config.extractors,
        "--ssaf-compilation-unit-id=" + build_artifact_id.get_path(),
        "--ssaf-tu-summary-file=" + summary,
    ]
    return summary, extra + shim.pass_config.ssaf_extra_clang_args


def _transformation_args(shim, build_artifact_id):
    ssaf_artifact_repo = shim.ssaf_artifact_repo
    edits = utils.make_parents(
        ssaf_artifact_repo.get_path_for_artifact(
            get_ssaf_artifact_id(build_artifact_id, SSAFArtifactKind.EDITS)
        )
    )
    extra = [
        "--ssaf-source-transformation=" + shim.pass_config.transformation,
        "--ssaf-global-scope-analysis-result="
        + config.wpa_result_path(shim.workflow_config.output_dir),
        "--ssaf-compilation-unit-id=" + build_artifact_id.get_path(),
        "--ssaf-link-unit-id=" + shim.pass_config.link_unit_id,
        "--ssaf-src-edit-file=" + edits,
        "--ssaf-transformation-report-file="
        + utils.make_parents(
            ssaf_artifact_repo.get_path_for_artifact(
                get_ssaf_artifact_id(build_artifact_id, SSAFArtifactKind.REPORT)
            )
        ),
    ]
    return edits, extra + shim.pass_config.ssaf_extra_clang_args


def _run_clang_driven_tool(shim, cwd):
    job = Job(list(shim.argv[1:]))

    if not job.output:
        logging_diagnostics.report_note(
            shim.workflow_config.verbose, "no output, no SSAF step to run: %r" % (job,)
        )
        return

    output = os.path.join(cwd, job.output)
    build_artifact_id = BuildArtifactId(output)
    if shim.pass_config.pass_kind == config.SSAFPassKind.TRANSFORMATION:
        artifact, extra = _transformation_args(shim, build_artifact_id)
        failure_kind = SSAFFailureKind.TRANSFORM
    else:
        artifact, extra = _summary_extraction_args(shim, build_artifact_id)
        failure_kind = SSAFFailureKind.EXTRACT
    ssaf_command = [shim.pass_config.ssaf_clang] + shim.argv[1:] + extra

    if not shim.run(ssaf_command, failure_kind, build_artifact_id):
        try:
            os.remove(artifact)
        except OSError:
            pass
        logging_diagnostics.report_error(
            "discarded incomplete artifact for %s" % build_artifact_id.get_path()
        )


def invoke_ssaf_tools(shim, cwd):
    if shim.skip_binary_tool_invocation:
        _run_clang_driven_tool(shim, cwd)
    else:
        _invoke_all_jobs(shim, shim.bin_tool_path(), cwd)


def main(argv, env=None):
    return run.run_shim(invoke_ssaf_tools, argv, env)
