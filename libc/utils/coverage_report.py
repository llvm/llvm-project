#!/usr/bin/env python3
#===-- coverage_report.py ------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

import argparse
import glob
import os
import shutil
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate libc code coverage report")
    parser.add_argument("--build-dir", required=True, help="Path to the libc build directory")
    parser.add_argument("--llvm-tools-dir", required=True, help="Path to the LLVM tools directory (llvm-profdata, llvm-cov)")
    args = parser.parse_args()

    build_dir = args.build_dir
    tools_dir = args.llvm_tools_dir

    profdata_tool = os.path.join(tools_dir, "llvm-profdata")
    cov_tool = os.path.join(tools_dir, "llvm-cov")

    if not os.path.isfile(profdata_tool):
        profdata_tool = shutil.which("llvm-profdata-19") or shutil.which("llvm-profdata")
    
    if not os.path.isfile(cov_tool):
        cov_tool = shutil.which("llvm-cov-19") or shutil.which("llvm-cov")

    if not profdata_tool or not cov_tool:
        print(f"Error: Could not find llvm-profdata or llvm-cov in {tools_dir} or in PATH")
        sys.exit(1)

    # Find all .profraw files recursively throughout the build tree
    # This ensures we catch profiles dumped in subdirectories if LLVM_PROFILE_FILE isn't strictly adhered to.
    profraw_files = glob.glob(os.path.join(build_dir, "**", "*.profraw"), recursive=True)

    if not profraw_files:
        print(f"Error: No .profraw files found in {build_dir}. Did tests run with coverage enabled?")
        sys.exit(1)

    # Prevent command line too long error by writing paths to a file
    list_file = os.path.join(build_dir, "profraw_list.txt")
    with open(list_file, "w") as f:
        for p in profraw_files:
            f.write(p + "\n")

    merged_profdata = os.path.join(build_dir, "merged.profdata")

    print(f"Merging {len(profraw_files)} profiles into {merged_profdata}...")
    subprocess.check_call(
        [profdata_tool, "merge", "-sparse", f"-input-files={list_file}", "-o", merged_profdata]
    )
    
    # Remove raw profiles after merge to save space and prevent inflation on next run
    for p in profraw_files:
        try:
            os.remove(p)
        except OSError:
            pass
    if os.path.exists(list_file):
        os.remove(list_file)

    test_dir = os.path.join(build_dir, "test")
    test_binaries = []
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.endswith(".__build__"):
                test_binaries.append(os.path.join(root, f))

    if not test_binaries:
        print(f"Error: No test binaries found in {test_dir}")
        sys.exit(1)

    # Use the first binary as the main object, the rest as -object arguments
    cov_cmd = [
        cov_tool, "report", test_binaries[0],
        f"-instr-profile={merged_profdata}",
        "--show-mcdc-summary"
    ]

    for tb in test_binaries[1:]:
        cov_cmd.append(f"-object={tb}")

    # Add source path filtering so we only get coverage for core library implementation,
    # rather than inflating coverage with test suite files.
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cov_cmd.extend([
        os.path.join(workspace_root, "src"),
        os.path.join(workspace_root, "include"),
        os.path.join(workspace_root, "hdr")
    ])

    print("Generating coverage report...")
    subprocess.check_call(cov_cmd)

if __name__ == "__main__":
    main()
