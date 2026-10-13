# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from .. import config, logging_diagnostics, utils
from ..build_graph import BuildArtifactId
from ..failure_logs import SSAFFailureKind
from ..ssaf_artifact_repository import SSAFArtifactKind, get_ssaf_artifact_id
from . import run
from .commands import (
    Files,
    filter_build_artifacts,
    is_object,
    summary_of,
    unique,
)
from .shim import merge_edits


def identify_inputs_output(argv):
    inputs = []
    output = None
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "-o" and index + 1 < len(argv):
            output = argv[index + 1]
            index += 2
            continue
        if not arg.startswith("-") and is_object(arg):
            inputs.append(arg)
        index += 1
    return Files(inputs, output) if inputs else None


def link_summaries(shim, build_artifact_id, input_build_artifact_ids):
    ssaf_artifact_repo = shim.ssaf_artifact_repo
    summaries = []
    for each in input_build_artifact_ids:
        found = summary_of(ssaf_artifact_repo, each)
        if found:
            summaries.append(found)
        else:
            shim.log_missing_summary(build_artifact_id, each)
    summaries = unique(summaries)
    if not summaries:
        logging_diagnostics.report_note(
            shim.workflow_config.verbose,
            "no summaries feed %s, nothing to link" % build_artifact_id.get_path(),
        )
        return
    target = utils.make_parents(
        ssaf_artifact_repo.get_path_for_artifact(
            get_ssaf_artifact_id(build_artifact_id, SSAFArtifactKind.SUMMARY)
        )
    )
    shim.run(
        [shim.pass_config.ssaf_linker] + summaries + ["-o", target],
        SSAFFailureKind.LINK,
        build_artifact_id,
    )


def invoke_ssaf_tools(shim, cwd):
    files = identify_inputs_output(shim.argv)
    if files is None:
        logging_diagnostics.report_note(
            shim.workflow_config.verbose, "nothing linked, no SSAF step to run"
        )
        return

    ssaf_artifact_repo = shim.ssaf_artifact_repo
    output = os.path.join(cwd, files.output or "a.out")
    build_artifact_id = BuildArtifactId(output)
    input_build_artifact_ids = filter_build_artifacts(
        shim.build_graph, ssaf_artifact_repo, cwd, files.inputs
    )

    if shim.pass_config.pass_kind == config.SSAFPassKind.TRANSFORMATION:
        merge_edits(shim, build_artifact_id, input_build_artifact_ids)
        return

    shim.build_graph.record_build_step(
        build_artifact_id,
        inputs=input_build_artifact_ids,
    )
    link_summaries(shim, build_artifact_id, input_build_artifact_ids)


def main(argv, env=None):
    return run.run_shim(invoke_ssaf_tools, argv, env)
