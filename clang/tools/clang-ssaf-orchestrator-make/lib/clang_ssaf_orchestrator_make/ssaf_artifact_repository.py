# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import enum
import os

from .utils import coalesce_root_path, find_with_suffix


class SSAFArtifactKind(enum.Enum):
    SUMMARY = ".summary.json"


class SSAFArtifactId:
    __slots__ = ("build_artifact_id", "ssaf_artifact_kind")

    def __init__(self, build_artifact_id, ssaf_artifact_kind):
        self.build_artifact_id = build_artifact_id
        self.ssaf_artifact_kind = ssaf_artifact_kind

    def get_build_artifact_id(self):
        return self.build_artifact_id

    def __repr__(self):
        return "SSAFArtifactId(%r, %r)" % (
            self.build_artifact_id,
            self.ssaf_artifact_kind,
        )


def get_ssaf_artifact_id(build_artifact_id, ssaf_artifact_kind):
    return SSAFArtifactId(build_artifact_id, ssaf_artifact_kind)


class SSAFArtifactRepository:
    def __init__(self, ssaf_artifact_dir):
        self.ssaf_artifact_dir = os.path.abspath(ssaf_artifact_dir)

    def get_build_artifact_path(self, ssaf_artifact_id):
        return ssaf_artifact_id.get_build_artifact_id().get_path()

    def get_path_for_artifact(self, ssaf_artifact_id):
        return coalesce_root_path(
            self.ssaf_artifact_dir,
            self.get_build_artifact_path(ssaf_artifact_id)
            + ssaf_artifact_id.ssaf_artifact_kind.value,
        )

    def has_artifact(self, ssaf_artifact_id):
        return os.path.exists(self.get_path_for_artifact(ssaf_artifact_id))

    def _all_paths_of_kind(self, ssaf_artifact_kind):
        return find_with_suffix(self.ssaf_artifact_dir, ssaf_artifact_kind.value)

    def count_summaries(self):
        return len(self._all_paths_of_kind(SSAFArtifactKind.SUMMARY))
