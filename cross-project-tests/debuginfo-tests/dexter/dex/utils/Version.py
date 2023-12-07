# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""DExTer version output."""

import os
from subprocess import CalledProcessError, check_output, STDOUT
import sys

from dex import __version__


def _git_version():
    dir_ = os.path.dirname(__file__)
    try:
        branch = (
            check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=STDOUT, cwd=dir_
            )
            .rstrip()
            .decode("utf-8")
        )
        hash_ = (
            check_output(["git", "rev-parse", "HEAD"], stderr=STDOUT, cwd=dir_)
            .rstrip()
            .decode("utf-8")
        )
        repo = (
            check_output(
                ["git", "remote", "get-url", "origin"], stderr=STDOUT, cwd=dir_
            )
            .rstrip()
            .decode("utf-8")
        )
        return "[{} {}] ({})".format(branch, hash_, repo)
    except (OSError, CalledProcessError):
        pass
    return None


def version(name):
    lines = []
    lines.append(" ".join([s for s in [name, __version__, _git_version()] if s]))
    lines.append("  using Python {}".format(sys.version))
    return "\n".join(lines)
