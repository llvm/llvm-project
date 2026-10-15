# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import re

from .options import ENVIRONMENT_ERROR, Problem

Replacement = collections.namedtuple("Replacement", "file_path offset length text")

_ENTRY_KEYS = ("Offset", "Length", "ReplacementText")

_LINE = re.compile(
    r"^(?P<indent>(?:  - |    )?)(?P<key>[A-Za-z]+):(?: +(?P<value>.*))?$"
)

_ESCAPE = re.compile(r"\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)")

_ESCAPED = {
    "0": "\0",
    "a": "\a",
    "b": "\b",
    "t": "\t",
    "n": "\n",
    "v": "\v",
    "f": "\f",
    "r": "\r",
    "e": "\x1b",
    '"': '"',
    "\\": "\\",
    "N": "\x85",
    "_": "\xa0",
    "L": "\u2028",
    "P": "\u2029",
}


class _Malformed(Exception):
    def __init__(self, line, message):
        super().__init__(message)
        self.line = line
        self.message = message


def _unescape(body, line):
    pieces = []
    position = 0
    for match in _ESCAPE.finditer(body):
        pieces.append(body[position : match.start()])
        code = match.group(1)
        if len(code) > 1:
            pieces.append(chr(int(code[1:], 16)))
        elif code in _ESCAPED:
            pieces.append(_ESCAPED[code])
        else:
            raise _Malformed(line, "unknown escape '\\%s'" % code)
        position = match.end()
    rest = body[position:]
    if "\\" in rest:
        raise _Malformed(line, "unfinished escape")
    pieces.append(rest)
    return "".join(pieces)


def _scalar(value, line):
    if value is None:
        raise _Malformed(line, "missing value")
    if value[:1] in ("'", '"'):
        if len(value) < 2 or value[-1] != value[0]:
            raise _Malformed(line, "unterminated string")
        if value[0] == "'":
            return value[1:-1].replace("''", "'")
        return _unescape(value[1:-1], line)
    return value


def _integer(value, line):
    text = _scalar(value, line)
    if not text.isdigit():
        raise _Malformed(line, "expected a number, got '%s'" % text)
    return int(text)


def _parse(lines):
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0] != "---":
        raise _Malformed(1, "expected '---'")
    if len(lines) < 2 or lines[-1] != "...":
        raise _Malformed(len(lines), "expected '...'")
    entries = []
    replacements_line = None
    for number, text in enumerate(lines[1:-1], start=2):
        match = _LINE.match(text)
        if not match:
            raise _Malformed(number, "unexpected line")
        indent, key, value = match.group("indent", "key", "value")
        if not indent and key == "MainSourceFile":
            _scalar(value, number)
        elif not indent and key == "Replacements" and replacements_line is None:
            replacements_line = number
            if value is not None and value != "[]":
                raise _Malformed(number, "expected a list of replacements")
        elif indent == "  - " and key == "FilePath" and replacements_line:
            entries.append((number, {key: _scalar(value, number)}))
        elif indent == "    " and key in _ENTRY_KEYS and entries:
            entry = entries[-1][1]
            if key in entry:
                raise _Malformed(number, "'%s' given twice" % key)
            if key == "ReplacementText":
                entry[key] = _scalar(value, number)
            else:
                entry[key] = _integer(value, number)
        else:
            raise _Malformed(number, "unexpected '%s'" % key)
    if replacements_line is None:
        raise _Malformed(len(lines), "no replacements")
    replacements = []
    for number, entry in entries:
        missing = [key for key in _ENTRY_KEYS if key not in entry]
        if missing:
            raise _Malformed(number, "replacement without '%s'" % missing[0])
        replacements.append(
            Replacement(
                entry["FilePath"],
                entry["Offset"],
                entry["Length"],
                entry["ReplacementText"],
            )
        )
    return replacements


def read_source_edits(path):
    try:
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return None, Problem(
            ENVIRONMENT_ERROR, "cannot read the source edits: %s" % error
        )
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    try:
        return _parse(lines), None
    except _Malformed as error:
        return None, Problem(
            ENVIRONMENT_ERROR,
            "malformed source edits: %s:%d: %s" % (path, error.line, error.message),
        )
