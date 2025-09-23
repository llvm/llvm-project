#!/usr/bin/env python3
"""
Usage: clang-explicit-module-build.py [-cas-path CAS_PATH] -- CLANG_COMMAND

The script builds with clang explicit module. If `-cas-path` is used, clang
explicit module build is going to enable compilation caching.
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        print("missing clang command")
        exit(1)
    dash_idx = argv.index("--")
    clang_args = argv[dash_idx + 1:]
    if len(clang_args) == 0:
        print("empty clang command")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-cas-path", required=False)
    args = parser.parse_args(argv[:dash_idx])

    clang_exe = clang_args[0]
    clang_scan_deps = os.path.join(
        os.path.dirname(clang_exe), "clang-scan-deps")

    scan_cmd = [clang_scan_deps]
    if args.cas_path is None:
        scan_cmd.extend(["-format", "experimental-full", "-o", "-", "--"])
    else:
        scan_cmd.extend(["-format", "experimental-include-tree-full",
                        "-cas-path", args.cas_path, "-o", "-", "--"])
    scan_cmd.extend(clang_args)
    scan_result = json.loads(subprocess.check_output(scan_cmd))

    # build module: assuming modules in reverse dependency order.
    for module in reversed(scan_result["modules"]):
        cmd = [clang_exe] + module["command-line"]
        print(*cmd)
        subprocess.check_call(cmd)

    # build tu: assuming only one TU.
    tu_cmd = [clang_exe] + \
        scan_result["translation-units"][0]["commands"][0]["command-line"]
    print(*tu_cmd)
    subprocess.check_call(tu_cmd)


if __name__ == "__main__":
    main()
