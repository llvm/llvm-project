# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import signal
import subprocess
import sys
import traceback

from .. import build_graph, config, failure_logs, logging_diagnostics, utils
from ..failure_logs import SSAFFailureKind
from ..ssaf_artifact_repository import (
    SSAFArtifactKind,
    SSAFArtifactRepository,
    get_ssaf_artifact_id,
)
from .commands import artifact_of, unique

SKIP_BINARY_TOOL_INVOCATION = "--ssaf-shim-skip-binary-tool-invocation"


def error_no_config():
    logging_diagnostics.report_error(
        "missing configuration file",
        [
            "%s is not set, so this is not an orchestrator run"
            % config.SSAF_CONFIG_ENV_VAR
        ],
    )
    return 1


class Shim:
    def __init__(self, workflow_config, pass_config, argv):
        self.workflow_config = workflow_config
        self.pass_config = pass_config
        argv = list(argv)
        self.skip_binary_tool_invocation = (
            len(argv) > 1 and argv[-1] == SKIP_BINARY_TOOL_INVOCATION
        )
        if self.skip_binary_tool_invocation:
            del argv[-1]
        self.argv = argv
        self.invoked_as = os.path.basename(self.argv[0])
        self._ssaf_artifact_repo = None
        self._build_graph = None

    @property
    def ssaf_artifact_repo(self):
        if self._ssaf_artifact_repo is None:
            self._ssaf_artifact_repo = SSAFArtifactRepository(
                config.ssaf_artifact_dir(self.workflow_config.output_dir)
            )
        return self._ssaf_artifact_repo

    @property
    def build_graph(self):
        if self._build_graph is None:
            self._build_graph = build_graph.BuildGraph(
                config.build_graph_dir(self.workflow_config.output_dir)
            )
        return self._build_graph

    @property
    def shim_link_dir(self):
        return config.shim_link_dir(self.workflow_config.output_dir)

    def log_failure(self, command, completed, ssaf_failure_kind, build_artifact_id):
        log_path = utils.make_parents(
            failure_logs.get_failure_log_path(
                self.workflow_config.output_dir,
                ssaf_failure_kind,
                command,
                build_artifact_id,
            )
        )
        with open(log_path, "wb") as handle:
            handle.write(("$ " + " ".join(command) + "\n").encode("utf-8"))
            handle.write(completed.stdout or b"")
            handle.write(completed.stderr or b"")
        logging_diagnostics.report_error(
            "SSAF step failed (exit %d), see %s" % (completed.returncode, log_path)
        )

    def log_missing_summary(self, build_artifact_id, input_build_artifact_id):
        output_dir = self.workflow_config.output_dir
        path = input_build_artifact_id.get_path()
        if failure_logs.has_failure_log_for(output_dir, input_build_artifact_id):
            logging_diagnostics.report_note(
                self.workflow_config.verbose,
                "no summary for %s; already reported" % path,
            )
            return
        message = (
            "no summary for %s, consumed by %s; nothing it was assembled from "
            "was analysed" % (path, build_artifact_id.get_path())
        )
        log = utils.make_parents(
            failure_logs.get_failure_log_path(
                output_dir, SSAFFailureKind.MISSING, self.argv, input_build_artifact_id
            )
        )
        with open(log, "w") as handle:
            handle.write(message + "\n")
        logging_diagnostics.report_error(message)

    def get_unmodified_env(self):
        result = dict(os.environ)
        result.pop(config.SSAF_CONFIG_ENV_VAR, None)
        result["PATH"] = self.workflow_config.restored_path_env_var
        return result

    def bin_tool_path(self):
        return shutil.which(
            self.invoked_as, path=self.workflow_config.restored_path_env_var
        )

    def run_bin_tool(self, bin_tool_path, cwd):
        if not bin_tool_path:
            logging_diagnostics.report_error(
                "cannot find the binary %s" % self.invoked_as
            )
            return 127
        command = [bin_tool_path] + self.argv[1:]
        logging_diagnostics.report_note(
            self.workflow_config.verbose, "binary: %s" % " ".join(command)
        )
        try:
            status = subprocess.call(command, cwd=cwd, env=self.get_unmodified_env())
        except OSError as error:
            logging_diagnostics.report_error("cannot run %s: %s" % (command[0], error))
            return 127
        if status < 0:
            return self._propagate_process_return_status(status)
        return status

    def _propagate_process_return_status(self, return_status):
        signal_number = -return_status
        try:
            signal.signal(signal_number, signal.SIG_DFL)
            os.kill(os.getpid(), signal_number)
        except OSError:
            pass
        return 128 + signal_number

    def run(self, command, ssaf_failure_kind, build_artifact_id, relay_output=False):
        logging_diagnostics.report_note(
            self.workflow_config.verbose, "ssaf: %s" % " ".join(command)
        )
        try:
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            if relay_output and completed.stdout:
                sys.stderr.write(completed.stdout.decode("utf-8", "replace"))
                sys.stderr.flush()
        except OSError as error:
            logging_diagnostics.report_error("cannot run %s: %s" % (command[0], error))
            return False
        if completed.returncode == 0:
            return True
        self.log_failure(command, completed, ssaf_failure_kind, build_artifact_id)
        return False

    def protect(self, work):
        try:
            work()
        except Exception:  # noqa: BLE001 - deliberately everything
            logging_diagnostics.report_error("internal error in %s" % self.invoked_as)
            traceback.print_exc(file=sys.stderr)
            try:
                path = utils.make_parents(
                    failure_logs.get_failure_log_path(
                        self.workflow_config.output_dir,
                        SSAFFailureKind.INTERNAL,
                        self.argv,
                        None,
                    )
                )
                with open(path, "w") as handle:
                    handle.write("%s\n\n" % " ".join(self.argv))
                    traceback.print_exc(file=handle)
            except Exception:
                logging_diagnostics.report_error("and its log could not be written")
        # A shim must never be the reason a build fails.
        return 0


def _input_artifacts(ssaf_artifact_repo, input_build_artifact_ids, ssaf_artifact_kind):
    found = []
    for each in input_build_artifact_ids:
        artifact = artifact_of(ssaf_artifact_repo, each, ssaf_artifact_kind)
        if artifact:
            found.append(artifact)
    return unique(found)


def _output_artifact(ssaf_artifact_repo, build_artifact_id, ssaf_artifact_kind):
    return utils.make_parents(
        ssaf_artifact_repo.get_path_for_artifact(
            get_ssaf_artifact_id(build_artifact_id, ssaf_artifact_kind)
        )
    )


def _report_nothing_to_merge(shim, what, build_artifact_id):
    logging_diagnostics.report_note(
        shim.workflow_config.verbose,
        "no %s feed %s, nothing to merge" % (what, build_artifact_id.get_path()),
    )


def merge_edits(shim, build_artifact_id, input_build_artifact_ids):
    ssaf_artifact_repo = shim.ssaf_artifact_repo
    found = _input_artifacts(
        ssaf_artifact_repo, input_build_artifact_ids, SSAFArtifactKind.EDITS
    )
    if not found:
        _report_nothing_to_merge(shim, "edits", build_artifact_id)
        return
    command = [shim.pass_config.ssaf_src_edit_merge] + found
    command += [
        "-o",
        _output_artifact(ssaf_artifact_repo, build_artifact_id, SSAFArtifactKind.EDITS),
    ]
    command += [
        "--sarif-conflicts-out="
        + ssaf_artifact_repo.get_path_for_artifact(
            get_ssaf_artifact_id(build_artifact_id, SSAFArtifactKind.CONFLICTS)
        )
    ]
    shim.run(command, SSAFFailureKind.MERGE, build_artifact_id)


def merge_transformation_reports(shim, build_artifact_id, input_build_artifact_ids):
    ssaf_artifact_repo = shim.ssaf_artifact_repo
    found = _input_artifacts(
        ssaf_artifact_repo, input_build_artifact_ids, SSAFArtifactKind.REPORT
    )
    if not found:
        _report_nothing_to_merge(shim, "transformation reports", build_artifact_id)
        return
    command = [shim.pass_config.ssaf_report_merge] + found
    command += [
        "-o",
        _output_artifact(
            ssaf_artifact_repo, build_artifact_id, SSAFArtifactKind.REPORT
        ),
    ]
    shim.run(command, SSAFFailureKind.TRANSFORMATION_REPORT, build_artifact_id)


def merge_conflict_reports(shim, build_artifact_id, input_build_artifact_ids):
    ssaf_artifact_repo = shim.ssaf_artifact_repo
    found = _input_artifacts(
        ssaf_artifact_repo, input_build_artifact_ids, SSAFArtifactKind.CONFLICT_REPORT
    )
    own = artifact_of(ssaf_artifact_repo, build_artifact_id, SSAFArtifactKind.CONFLICTS)
    if own:
        found.append(own)
    if not found:
        _report_nothing_to_merge(shim, "conflict reports", build_artifact_id)
        return
    command = [shim.pass_config.ssaf_conflict_merge] + found
    command += [
        "-o",
        _output_artifact(
            ssaf_artifact_repo, build_artifact_id, SSAFArtifactKind.CONFLICT_REPORT
        ),
    ]
    shim.run(command, SSAFFailureKind.CONFLICT_REPORT, build_artifact_id)


def merge_transformation_outputs(shim, build_artifact_id, input_build_artifact_ids):
    merge_edits(shim, build_artifact_id, input_build_artifact_ids)
    merge_transformation_reports(shim, build_artifact_id, input_build_artifact_ids)
    merge_conflict_reports(shim, build_artifact_id, input_build_artifact_ids)
