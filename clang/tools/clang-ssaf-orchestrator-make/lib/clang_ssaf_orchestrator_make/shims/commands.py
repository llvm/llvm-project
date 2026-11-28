# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import os

from ..build_graph import BuildArtifactId
from ..ssaf_artifact_repository import SSAFArtifactKind, get_ssaf_artifact_id

Files = collections.namedtuple("Files", "inputs output")

_OBJECT_EXTENSIONS = frozenset([".o", ".obj"])
_ARCHIVE_EXTENSIONS = frozenset([".a"])


def is_object(filename):
    return os.path.splitext(filename)[1] in _OBJECT_EXTENSIONS


def is_archive(filename):
    return os.path.splitext(filename)[1] in _ARCHIVE_EXTENSIONS


def artifact_of(ssaf_artifact_repo, build_artifact_id, ssaf_artifact_kind):
    return ssaf_artifact_repo.get_path_if_present(
        get_ssaf_artifact_id(build_artifact_id, ssaf_artifact_kind)
    )


def summary_of(ssaf_artifact_repo, build_artifact_id):
    return artifact_of(ssaf_artifact_repo, build_artifact_id, SSAFArtifactKind.SUMMARY)


def filter_build_artifacts(build_graph, ssaf_artifact_repo, cwd, files):
    kept = []
    for each in files:
        build_artifact_id = BuildArtifactId(os.path.join(cwd, each))
        if (
            artifact_of(ssaf_artifact_repo, build_artifact_id, SSAFArtifactKind.SUMMARY)
            or artifact_of(
                ssaf_artifact_repo, build_artifact_id, SSAFArtifactKind.EDITS
            )
            or build_graph.was_built(build_artifact_id)
        ):
            kept.append(build_artifact_id)
    return kept


def unique(items):
    seen = set()
    kept = []
    for item in items:
        if item not in seen:
            seen.add(item)
            kept.append(item)
    return kept
