# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum
import os

SSAF_CONFIG_ENV_VAR = "SSAF_CONFIG"

SHIM_LINK_DIR = "shims"
BUILD_GRAPH_DIR = "build-graph"
SSAF_ARTIFACT_DIR = "ssaf-artifacts"
FAILURE_LOG_DIR = "failure-logs"
WPA_RESULT_FILE = "wpa.json"
OUTPUT_DIR_MARKER = ".clang-ssaf-orchestrator-make"


def shim_link_dir(output_dir):
    return os.path.join(output_dir, SHIM_LINK_DIR)


def build_graph_dir(output_dir):
    return os.path.join(output_dir, BUILD_GRAPH_DIR)


def ssaf_artifact_dir(output_dir):
    return os.path.join(output_dir, SSAF_ARTIFACT_DIR)


def failure_log_dir(output_dir):
    return os.path.join(output_dir, FAILURE_LOG_DIR)


def wpa_result_path(output_dir):
    return os.path.join(output_dir, WPA_RESULT_FILE)


def output_dir_marker_path(output_dir):
    return os.path.join(output_dir, OUTPUT_DIR_MARKER)


SHIM_FOR_LINK_NAME = {
    "cc": "clang",
    "clang": "clang",
    "c++": "clang",
    "clang++": "clang",
    "ld": "ld",
}


class SSAFPassKind(str, enum.Enum):
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"


class SSAFPassConfig:
    def __init__(self, ssaf_clang=None, ssaf_extra_clang_args=()):
        self.ssaf_clang = ssaf_clang
        self.ssaf_extra_clang_args = list(ssaf_extra_clang_args)

    @property
    def file_name(self):
        return "config-%s.json" % self.pass_kind.value

    def to_json(self):
        fields = dict(vars(self))
        fields["pass_kind"] = self.pass_kind.value
        return fields


class SSAFAnalysisPassConfig(SSAFPassConfig):
    pass_kind = SSAFPassKind.ANALYSIS

    def __init__(
        self,
        ssaf_clang=None,
        ssaf_extra_clang_args=(),
        extractors=None,
        ssaf_linker=None,
        analyses=(),
    ):
        super().__init__(ssaf_clang, ssaf_extra_clang_args)
        self.extractors = extractors
        self.ssaf_linker = ssaf_linker
        self.analyses = list(analyses)


class SSAFTransformationPassConfig(SSAFPassConfig):
    pass_kind = SSAFPassKind.TRANSFORMATION

    def __init__(
        self,
        ssaf_clang=None,
        ssaf_extra_clang_args=(),
        ssaf_src_edit_merge=None,
        transformation=None,
        link_unit_id=None,
    ):
        super().__init__(ssaf_clang, ssaf_extra_clang_args)
        self.ssaf_src_edit_merge = ssaf_src_edit_merge
        self.transformation = transformation
        self.link_unit_id = link_unit_id

    def name_link_unit(self, link_unit_id):
        self.link_unit_id = link_unit_id


class SSAFWorkflowConfig:
    def __init__(
        self,
        output_dir=None,
        installed_shim_dir_path=None,
        restored_path_env_var="",
        verbose=False,
        build_command=(),
        ssaf_analyzer=None,
        keep_going=False,
        build_environment=None,
    ):
        self.output_dir = output_dir
        self.installed_shim_dir_path = installed_shim_dir_path
        self.restored_path_env_var = restored_path_env_var
        self.verbose = verbose
        self.build_command = list(build_command)
        self.ssaf_analyzer = ssaf_analyzer
        self.keep_going = keep_going
        self.build_environment = dict(build_environment or {})

    @classmethod
    def written_field_names(cls):
        return frozenset(cls().to_json())

    def to_json(self):
        fields = dict(vars(self))
        del fields["build_environment"]
        return fields
