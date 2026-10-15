# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum
import hashlib
import os

from . import config
from .utils import coalesce_root_path, find_with_suffix

FAILURE_LOG_SUFFIX = ".log"


class SSAFFailureKind(enum.Enum):
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LINK = "link"
    MERGE = "merge"
    FORWARD = "forward"
    MISSING = "missing"
    DRIVER_JOBS = "driver-jobs"
    INTERNAL = "internal"


def _command_hash(command):
    joined = "\0".join(command).encode("utf-8", "surrogateescape")
    return hashlib.sha1(joined).hexdigest()[:12]


def get_failure_log_path(output_dir, ssaf_failure_kind, command, build_artifact_id):
    if build_artifact_id is None:
        name = os.path.join(os.getcwd(), _command_hash(command))
    else:
        name = build_artifact_id.get_path()
    return coalesce_root_path(
        config.failure_log_dir(output_dir),
        "%s.%s%s" % (name, ssaf_failure_kind.value, FAILURE_LOG_SUFFIX),
    )


def has_failure_log_for(output_dir, build_artifact_id):
    return any(
        os.path.exists(get_failure_log_path(output_dir, each, None, build_artifact_id))
        for each in SSAFFailureKind
    )


def count_failure_logs(output_dir):
    return len(find_with_suffix(output_dir, FAILURE_LOG_SUFFIX))
