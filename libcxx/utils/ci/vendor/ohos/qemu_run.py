#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _parse_kv(items):
    result = {}
    for item in items:
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _resolve_binary(binary):
    if os.path.isabs(binary):
        if os.path.isfile(binary) and os.access(binary, os.X_OK):
            return binary
        sys.exit(f"failed to find executable: {binary}")

    resolved = shutil.which(binary)
    if resolved is None:
        sys.exit(f"failed to find executable: {binary}")
    return resolved


def _find_build_root(execdir):
    current = Path(execdir).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "CMakeCache.txt").exists():
            return candidate
    return None


def _collect_build_library_dirs(execdir):
    build_root = _find_build_root(execdir)
    if build_root is None:
        return []

    lib_root = build_root / "lib"
    if not lib_root.is_dir():
        return []

    library_dirs = []
    seen = set()
    for lib in sorted(lib_root.rglob("*.so*")):
        if not lib.is_file():
            continue
        parent = lib.parent.resolve()
        key = str(parent)
        if key in seen:
            continue
        seen.add(key)
        library_dirs.append(str(parent))
    return library_dirs


def _build_target_env_args(env, prepend_env):
    target_env = dict(env)

    for key, value in prepend_env.items():
        existing = target_env.get(key, os.environ.get(key, ""))
        target_env[key] = f"{value}{os.pathsep}{existing}" if existing else value

    args = []
    for key, value in target_env.items():
        args.extend(["-E", f"{key}={value}"])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qemu",
        default=os.environ.get("OHOS_QEMU", os.environ.get("QEMU", "qemu-aarch64")),
    )
    parser.add_argument("--sysroot", required=True)
    parser.add_argument("--execdir", required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(
            os.environ.get(
                "OHOS_QEMU_TIMEOUT_SECONDS",
                os.environ.get("QEMU_TIMEOUT_SECONDS", "600"),
            )
        ),
    )
    parser.add_argument("--env", nargs="*", default=[])
    parser.add_argument("--prepend_env", nargs="*", default=[])
    parser.add_argument("command", nargs=argparse.ONE_OR_MORE)
    args = parser.parse_args()

    qemu = _resolve_binary(args.qemu)
    execdir = Path(args.execdir).resolve()
    env = _parse_kv(args.env)
    prepend_env = _parse_kv(args.prepend_env)

    build_library_dirs = _collect_build_library_dirs(execdir)
    if build_library_dirs:
        existing = prepend_env.get("LD_LIBRARY_PATH")
        build_library_path = os.pathsep.join(build_library_dirs)
        prepend_env["LD_LIBRARY_PATH"] = (
            f"{build_library_path}{os.pathsep}{existing}"
            if existing
            else build_library_path
        )

    cmd = [
        qemu,
        "-L",
        args.sysroot,
        *_build_target_env_args(env, prepend_env),
        *args.command,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=execdir,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"qemu executor timed out after {args.timeout_seconds}s: {' '.join(args.command)}",
            file=sys.stderr,
        )
        return 124
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
