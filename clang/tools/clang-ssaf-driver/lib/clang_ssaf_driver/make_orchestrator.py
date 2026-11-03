# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys

from . import logging_diagnostics

ORCHESTRATOR = "clang-ssaf-orchestrator-make"

RESULT_KEYS = frozenset(["source-edits", "transformation-report", "conflict-report"])


def find_orchestrator():
    package_dir = os.path.dirname(os.path.realpath(__file__))
    prefix = os.path.dirname(os.path.dirname(package_dir))
    return os.path.join(prefix, "bin", ORCHESTRATOR)


def orchestrator_command(orchestrator, transformation, command_line_opts):
    return (
        [
            sys.executable,
            orchestrator,
            "--extract-summaries=" + transformation.extract_summaries,
            "--analysis=" + transformation.analysis,
            "--transformation=" + transformation.transformation,
            "--output-dir=" + command_line_opts.output_dir,
        ]
        + command_line_opts.orchestrator_args
        + ["--"]
        + command_line_opts.build_command
    )


def exit_status(return_code):
    return 128 - return_code if return_code < 0 else return_code


def read_result(line):
    key, separator, value = line.rstrip(b"\n").partition(b": ")
    key = key.decode("ascii", "replace")
    if not separator or key not in RESULT_KEYS:
        return None
    return key, os.fsdecode(value)


def run_orchestrator(command, verbose, relay=None):
    relay = relay or sys.stdout.buffer
    logging_diagnostics.report_note(verbose, "running: %s" % " ".join(command))
    results = {}
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        try:
            for line in process.stdout:
                relay.write(line)
                relay.flush()
                result = read_result(line)
                if result:
                    results[result[0]] = result[1]
        except KeyboardInterrupt:
            pass
    return exit_status(process.returncode), results
