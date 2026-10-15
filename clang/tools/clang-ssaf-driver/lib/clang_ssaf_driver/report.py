# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import json
import os
import urllib.parse

from . import PROGRAM
from .options import ENVIRONMENT_ERROR, Problem

SCHEMA = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/"
    "sarif-schema-2.1.0.json"
)

_LEVELS = {"error": "error", "warning": "warning", "note": "note", "none": "note"}

_FILE_SCHEME = "file://"


def load(paths):
    runs = []
    for path in paths:
        if not path:
            continue
        try:
            with open(path) as handle:
                document = json.load(handle)
        except (OSError, ValueError) as error:
            return None, Problem(
                ENVIRONMENT_ERROR, "cannot read the report %s: %s" % (path, error)
            )
        if not isinstance(document, dict) or not isinstance(document.get("runs"), list):
            return None, Problem(ENVIRONMENT_ERROR, "not a SARIF log: %s" % path)
        runs.extend(document["runs"])
    return {"$schema": SCHEMA, "version": "2.1.0", "runs": runs}, None


def write(path, log):
    with open(path, "w") as handle:
        json.dump(log, handle, indent=2, sort_keys=True)
        handle.write("\n")


class _Sources:
    def __init__(self):
        self.contents = {}

    def position(self, path, offset):
        if path not in self.contents:
            try:
                with open(path, "rb") as handle:
                    self.contents[path] = handle.read()
            except OSError:
                self.contents[path] = None
        content = self.contents[path]
        if content is None or offset > len(content):
            return None
        line_start = content.rfind(b"\n", 0, offset) + 1
        return content.count(b"\n", 0, offset) + 1, offset - line_start + 1


def _path(uri):
    if not uri.startswith(_FILE_SCHEME):
        return uri
    return urllib.parse.unquote(uri[len(_FILE_SCHEME) :])


def _where(location, project_root, sources):
    physical = location.get("physicalLocation") or {}
    uri = (physical.get("artifactLocation") or {}).get("uri")
    if not uri:
        return PROGRAM
    path = _path(uri)
    shown = os.path.relpath(path, project_root)
    if shown == os.pardir or shown.startswith(os.pardir + os.sep):
        shown = path
    region = physical.get("region") or {}
    if "startLine" in region:
        return "%s:%d:%d" % (shown, region["startLine"], region.get("startColumn", 1))
    if "byteOffset" in region:
        position = sources.position(path, region["byteOffset"])
        if position:
            return "%s:%d:%d" % ((shown,) + position)
    return shown


def _text(item):
    return (item.get("message") or {}).get("text", "")


def diagnostics(log, project_root):
    sources = _Sources()
    lines = []
    counts = collections.Counter()
    for run in log["runs"]:
        for result in run.get("results") or []:
            rule_id = result.get("ruleId", "")
            counts[rule_id] += 1
            locations = result.get("locations") or [{}]
            lines.append(
                "%s: %s: %s [%s]"
                % (
                    _where(locations[0], project_root, sources),
                    _LEVELS.get(result.get("level"), "warning"),
                    _text(result),
                    rule_id,
                )
            )
            for related in result.get("relatedLocations") or []:
                lines.append(
                    "%s: note: %s"
                    % (_where(related, project_root, sources), _text(related))
                )
    if counts:
        summary = ["%d %s" % (counts[rule_id], rule_id) for rule_id in sorted(counts)]
        lines.append("%s reported" % ", ".join(summary))
    return lines


def conflict_ranges(log):
    ranges = {}
    for run in log["runs"]:
        for result in run.get("results") or []:
            for related in result.get("relatedLocations") or []:
                physical = related.get("physicalLocation") or {}
                uri = (physical.get("artifactLocation") or {}).get("uri")
                region = physical.get("region") or {}
                if uri and "byteOffset" in region and "byteLength" in region:
                    start = region["byteOffset"]
                    ranges.setdefault(_path(uri), []).append(
                        (start, start + region["byteLength"])
                    )
    return ranges
