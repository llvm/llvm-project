#!/usr/bin/env python3
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

import argparse
import os
import shutil
import subprocess
import tempfile

# The purpose of this script is to measure the performance effect
# of an lld change in a statistically sound way, automating all the
# tedious parts of doing so. It copies the test case into /tmp as well as
# running the test binaries from /tmp to reduce the influence on the test
# machine's storage medium on the results. It accounts for measurement
# bias caused by binary layout (using the --randomize-section-padding
# flag to link the test binaries) and by environment variable size
# (implemented by hyperfine [1]). Runs of the base and test case are
# interleaved to account for environmental factors which may influence
# the result due to the passage of time. The results of running hyperfine
# are collected into a results.csv file in the output directory and may
# be analyzed by the user with a tool such as ministat.
#
# Requirements: Linux host, hyperfine [2] in $PATH, run from a build directory
# configured to use ninja and a recent version of lld that supports
# --randomize-section-padding, /tmp is tmpfs.
#
# [1] https://github.com/sharkdp/hyperfine/blob/3cedcc38d0c430cbf38b4364b441c43a938d2bf3/src/util/randomized_environment_offset.rs#L1
# [2] https://github.com/sharkdp/hyperfine
#
# Example invocation for comparing the performance of the current commit
# against the previous commit which is treated as the baseline, without
# linking debug info:
#
# lld/utils/run_benchmark.py \
#   --base-commit HEAD^ \
#   --test-commit HEAD \
#   --test-case lld/utils/speed-test-reproducers/result/firefox-x64/response.txt \
#   --num-iterations 512 \
#   --num-binary-variants 16 \
#   --output-dir outdir \
#   --ldflags=-S
#
# Then this bash command will compare the real time of the base and test cases.
#
# ministat -A \
#   <(grep lld-base outdir/results.csv | cut -d, -f2) \
#   <(grep lld-test outdir/results.csv | cut -d, -f2)

# We don't want to copy stat() information when we copy the reproducer
# to the temporary directory. Files in the Nix store are read-only so this will
# cause trouble when the linker writes the output file and when we want to clean
# up the temporary directory. Python doesn't provide a way to disable copying
# stat() information in shutil.copytree so we just monkeypatch shutil.copystat
# to do nothing.
shutil.copystat = lambda *args, **kwargs: 0

parser = argparse.ArgumentParser(prog="benchmark_change.py")
parser.add_argument("--base-commit", required=True)
parser.add_argument("--test-commit", required=True)
parser.add_argument("--test-case", required=True)
parser.add_argument("--num-iterations", type=int, required=True)
parser.add_argument("--num-binary-variants", type=int, required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--ldflags", required=False)
args = parser.parse_args()

test_dir = tempfile.mkdtemp()
print(f"Using {test_dir} as temporary directory")

os.makedirs(args.output_dir)
print(f"Using {args.output_dir} as output directory")


def extract_link_command(target):
    # We assume that the last command printed by "ninja -t commands" containing a
    # "-o" flag is the link command (we need to check for -o because subsequent
    # commands create symlinks for ld.lld and so on). This is true for CMake and
    # gn.
    link_command = None
    for line in subprocess.Popen(
        ["ninja", "-t", "commands", target], stdout=subprocess.PIPE
    ).stdout.readlines():
        commands = line.decode("utf-8").split("&&")
        for command in commands:
            if " -o " in command:
                link_command = command.strip()
    return link_command


def generate_binary_variants(case_name):
    subprocess.run(["ninja", "lld"])
    link_command = extract_link_command("lld")

    for i in range(0, args.num_binary_variants):
        print(f"Generating binary variant {i} for {case_name} case")
        command = f"{link_command} -o {test_dir}/lld-{case_name}{i} -Wl,--randomize-section-padding={i}"
        subprocess.run(command, check=True, shell=True)


# Make sure that there are no local changes.
subprocess.run(["git", "diff", "--exit-code", "HEAD"], check=True)

# Resolve the base and test commit, since if they are relative to HEAD we will
# check out the wrong commit below.
resolved_base_commit = subprocess.check_output(
    ["git", "rev-parse", args.base_commit]
).strip()
resolved_test_commit = subprocess.check_output(
    ["git", "rev-parse", args.test_commit]
).strip()

test_case_dir = os.path.dirname(args.test_case)
test_case_respfile = os.path.basename(args.test_case)

test_dir_test_case_dir = f"{test_dir}/testcase"
shutil.copytree(test_case_dir, test_dir_test_case_dir)

subprocess.run(["git", "checkout", resolved_base_commit], check=True)
generate_binary_variants("base")

subprocess.run(["git", "checkout", resolved_test_commit], check=True)
generate_binary_variants("test")


def hyperfine_link_command(case_name):
    return f'../lld-{case_name}$(({{iter}}%{args.num_binary_variants})) -flavor ld.lld @{test_case_respfile} {args.ldflags or ""}'


results_csv = f"{args.output_dir}/results.csv"
subprocess.run(
    [
        "hyperfine",
        "--export-csv",
        os.path.abspath(results_csv),
        "-P",
        "iter",
        "0",
        str(args.num_iterations - 1),
        hyperfine_link_command("base"),
        hyperfine_link_command("test"),
    ],
    check=True,
    cwd=test_dir_test_case_dir,
)

shutil.rmtree(test_dir)
