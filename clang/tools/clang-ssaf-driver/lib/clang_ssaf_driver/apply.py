# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess

from . import logging_diagnostics, source_edits
from .make_orchestrator import exit_status
from .options import ENVIRONMENT_ERROR, Problem

TOOL = "clang-apply-replacements"
STAGING_DIR = "apply"
STAGED_EDITS = "source-edits.yaml"


def resolve_tool(cl_option_value):
    name_or_path = TOOL if cl_option_value is None else cl_option_value
    if os.sep in name_or_path:
        candidate = os.path.abspath(name_or_path)
    else:
        candidate = shutil.which(name_or_path)
    if not candidate:
        error = "%s: not found on PATH" % name_or_path
    elif not os.path.isfile(candidate):
        error = "not a file: %s" % candidate
    elif not os.access(candidate, os.X_OK):
        error = "not executable: %s" % candidate
    else:
        return candidate, None
    return None, Problem(
        ENVIRONMENT_ERROR,
        "--apply needs %s:" % TOOL,
        ["--clang-apply-replacements: " + error],
    )


def apply_source_edits(tool, files, output_dir, verbose):
    staging_dir = os.path.join(output_dir, STAGING_DIR)
    replacements = [
        source_edits.Replacement(each.path, offset, length, text)
        for each in files
        for offset, length, text in each.edits
    ]
    try:
        shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir)
        source_edits.write_source_edits(
            os.path.join(staging_dir, STAGED_EDITS), replacements
        )
    except OSError as error:
        return Problem(ENVIRONMENT_ERROR, "cannot stage the source edits: %s" % error)

    command = [tool, staging_dir]
    logging_diagnostics.report_note(verbose, "running: %s" % " ".join(command))
    try:
        status = exit_status(subprocess.call(command))
    except OSError as error:
        return Problem(127, "cannot run '%s': %s" % (tool, error))
    except KeyboardInterrupt:
        return Problem(130, "%s was interrupted" % TOOL)
    if status != 0:
        return Problem(status, "%s failed (exit %d)" % (TOOL, status))

    mismatched = []
    for each in files:
        try:
            with open(each.path, "rb") as handle:
                written = handle.read()
        except OSError:
            written = None
        if written != each.edited:
            mismatched.append(each.relative_path)
    if mismatched:
        return Problem(
            ENVIRONMENT_ERROR,
            "%s did not write what the patch shows:" % TOOL,
            mismatched,
        )
    return None
