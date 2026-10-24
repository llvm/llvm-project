# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
from unittest import mock

from clang_ssaf_orchestrator_make import build_graph, config, logging_diagnostics
from clang_ssaf_orchestrator_make import ssaf_artifact_repository as repository


def analysis_config(**overrides):
    fields = dict(
        ssaf_clang="/bin/clang",
        ssaf_extra_clang_args=[],
        extractors="PointerFlow",
        ssaf_linker="/bin/clang-ssaf-linker",
        analyses=[],
    )
    fields.update(overrides)
    return config.SSAFAnalysisPassConfig(**fields)


def transformation_config(**overrides):
    fields = dict(
        ssaf_clang="/bin/clang",
        ssaf_extra_clang_args=[],
        ssaf_src_edit_merge="/bin/clang-ssaf-src-edit-merge",
        transformation="x",
    )
    fields.update(overrides)
    return config.SSAFTransformationPassConfig(**fields)


def pass_configs():
    return analysis_config(), transformation_config(link_unit_id="build/prg")


def workflow_config(env=None, output_dir="/out", verbose=False):
    env = env or {}
    return config.SSAFWorkflowConfig(
        output_dir=output_dir,
        installed_shim_dir_path="/libexec",
        restored_path_env_var=env.get("PATH", ""),
        verbose=verbose,
        build_command=["make"],
        ssaf_analyzer="/bin/clang-ssaf-analyzer",
        keep_going=False,
        build_environment=env,
    )


@contextlib.contextmanager
def recorded_notes():
    notes = []
    with mock.patch.object(
        logging_diagnostics,
        "report_note",
        lambda verbose, message: notes.append(message),
    ):
        yield notes


class RecordingShim:
    """A Shim whose SSAF invocations are recorded rather than run."""

    def __init__(self, argv, output_dir, pass_config=None):
        from clang_ssaf_orchestrator_make.shims import shim as shim_module

        self.commands = []
        self.relayed = []
        self.missing = []
        self._shim = shim_module.Shim(
            workflow_config(output_dir=output_dir),
            pass_config or analysis_config(),
            argv,
        )
        self._shim.run = self._run
        self._shim.log_missing_summary = (
            lambda build_artifact_id, input_build_artifact_id: self.missing.append(
                input_build_artifact_id.get_path()
            )
        )

    def _run(self, command, ssaf_failure_kind, build_artifact_id, relay_output=False):
        self.commands.append(((ssaf_failure_kind, build_artifact_id), command))
        self.relayed.append(relay_output)
        return True

    def __getattr__(self, name):
        return getattr(self._shim, name)


def repository_for(output_dir):
    return repository.SSAFArtifactRepository(config.ssaf_artifact_dir(output_dir))


def build_graph_for(output_dir):
    return build_graph.BuildGraph(config.build_graph_dir(output_dir))
