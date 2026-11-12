# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import errno
import json
import os

from . import logging_diagnostics


def coalesce_root_path(root, path):
    coalesced = os.path.normpath(os.path.join(root, path.lstrip(os.sep)))
    if coalesced != root and not coalesced.startswith(root + os.sep):
        raise ValueError("path leaves %s: %s" % (root, coalesced))
    return coalesced


def find_with_suffix(root, suffix):
    found = []
    for directory, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(suffix):
                found.append(os.path.join(directory, name))
    return found


def make_parents(path):
    directory = os.path.dirname(path)
    if directory:
        try:
            os.makedirs(directory)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
    return path


def read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except OSError:
        return None
    except ValueError as error:
        logging_diagnostics.report_error("%s is not valid JSON: %s" % (path, error))
        return None


def write_json(path, document):
    temporary = "%s.%d.tmp" % (path, os.getpid())
    with open(temporary, "w") as handle:
        json.dump(document, handle, indent=2)
    os.replace(temporary, path)
