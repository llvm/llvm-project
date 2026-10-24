# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import difflib
import os

from .options import ENVIRONMENT_ERROR, Problem

FileEdits = collections.namedtuple(
    "FileEdits", "relative_path path edits original edited"
)

Collected = collections.namedtuple("Collected", "files outside blocked")

NO_NEWLINE = b"\\ No newline at end of file\n"


def apply_edits(original, edits):
    pieces = []
    position = 0
    previous = None
    for offset, length, text in edits:
        if offset + length > len(original):
            return None, "an edit at byte %d runs past the end of the file" % offset
        if offset < position or (length == 0 and previous == (offset, 0)):
            return None, "edits overlap at byte %d" % offset
        pieces.append(original[position:offset])
        pieces.append(text.encode("utf-8"))
        position = offset + length
        previous = (offset, length)
    pieces.append(original[position:])
    return b"".join(pieces), None


def _identity(path):
    try:
        status = os.stat(path)
    except OSError:
        return path
    return (status.st_dev, status.st_ino)


def _overlaps(edit, start, end):
    offset, length, _ = edit
    if length == 0:
        return start < offset < end
    return offset < end and start < offset + length


def collect(replacements, project_root, conflicts):
    grouped = {}
    paths = {}
    for each in replacements:
        path = os.path.realpath(os.path.join(project_root, each.file_path))
        identity = _identity(path)
        paths[identity] = min(paths.get(identity, path), path)
        grouped.setdefault(identity, set()).add((each.offset, each.length, each.text))
    conflicting = {}
    for path, ranges in conflicts.items():
        conflicting.setdefault(_identity(os.path.realpath(path)), []).extend(ranges)

    files = []
    outside = []
    blocked = []
    problems = []
    for identity in sorted(grouped, key=lambda each: paths[each]):
        path = paths[identity]
        relative_path = os.path.relpath(path, project_root)
        if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
            outside.append(path)
            continue
        ranges = conflicting.get(identity, ())
        edits = sorted(
            (
                edit
                for edit in grouped[identity]
                if not any(_overlaps(edit, start, end) for start, end in ranges)
            ),
            key=lambda edit: (edit[0], edit[1]),
        )
        if len(edits) < len(grouped[identity]):
            blocked.append((relative_path, len(grouped[identity]) - len(edits)))
        if not edits:
            continue
        try:
            with open(path, "rb") as handle:
                original = handle.read()
        except OSError as error:
            problems.append("%s: %s" % (relative_path, error.strerror))
            continue
        edited, problem = apply_edits(original, edits)
        if problem:
            problems.append("%s: %s" % (relative_path, problem))
            continue
        files.append(FileEdits(relative_path, path, edits, original, edited))

    if problems:
        return None, Problem(
            ENVIRONMENT_ERROR, "the source edits cannot be applied:", problems
        )
    return Collected(files, outside, blocked), None


def _lines(content):
    lines = content.split(b"\n")
    last = lines.pop()
    lines = [line + b"\n" for line in lines]
    if last:
        lines.append(last)
    return lines


def render(files):
    pieces = []
    for each in files:
        name = os.fsencode(each.relative_path)
        for line in difflib.diff_bytes(
            difflib.unified_diff,
            _lines(each.original),
            _lines(each.edited),
            b"a/" + name,
            b"b/" + name,
        ):
            pieces.append(line)
            if not line.endswith(b"\n"):
                pieces.append(b"\n" + NO_NEWLINE)
    return b"".join(pieces)
