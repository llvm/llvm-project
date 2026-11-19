# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

_JOB_PREFIX = b' "'

_ESCAPE_PAIR = re.compile(rb'\\[\\"$]')
_ARGUMENT = re.compile(rb'"((?:[^"\\]|\\.)*)"', re.DOTALL)
_UNESCAPE = re.compile(rb'\\(["\\$])')


def _quotes_balanced(record):
    return _ESCAPE_PAIR.sub(b"", record).count(b'"') % 2 == 0


def _records(data):
    lines = data.split(b"\n")
    found = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(_JOB_PREFIX):
            record = line
            while not _quotes_balanced(record) and index + 1 < len(lines):
                index += 1
                record += b"\n" + lines[index]
            found.append(record)
        index += 1
    return found


def _tokenize(record):
    return [
        os.fsdecode(_UNESCAPE.sub(rb"\1", match.group(1)))
        for match in _ARGUMENT.finditer(record)
    ]


class Job:
    def __init__(self, argv):
        self.argv = argv

    @property
    def output(self):
        for index, arg in enumerate(self.argv):
            if arg == "-o" and index + 1 < len(self.argv):
                return self.argv[index + 1]
        return None

    def __repr__(self):
        return "Job(argv=%r, output=%r)" % (self.argv, self.output)


def parse_driver_job_listing(data):
    return [Job(_tokenize(record)) for record in _records(data)]
