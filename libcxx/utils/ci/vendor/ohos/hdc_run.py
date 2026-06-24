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
import re
import shlex
import shutil
import subprocess
import sys
import uuid


EXIT_MARKER = "__llvm_hdc_exit__="


def _parse_kv(items):
    result = {}
    for item in items:
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _split_path_list(value):
    return [entry for entry in value.split(os.pathsep) if entry]


def _resolve_hdc_binary(hdc):
    if os.path.isabs(hdc):
        return hdc
    resolved = shutil.which(hdc)
    if resolved is None:
        sys.exit(f"failed to find hdc binary: {hdc}")
    return resolved


def _is_windows_hdc(path):
    return path.lower().endswith(".exe")


def _find_windows_local_root(hdc_path):
    parts = Path(hdc_path).parts
    for index in range(len(parts) - 1):
        if parts[index] == "AppData" and index + 1 < len(parts) and parts[index + 1] == "Local":
            return Path(*parts[: index + 2])
    return None


def _default_host_staging_root(hdc_path):
    if _is_windows_hdc(hdc_path):
        local_root = _find_windows_local_root(hdc_path)
        if local_root is None:
            sys.exit(
                "failed to derive a Windows staging directory from the hdc.exe path; "
                "set HDC_HOST_STAGING_ROOT to a path under /mnt/<drive>/..."
            )
        return local_root / "Temp" / "llvm-lit-hdc"
    return Path("/tmp/llvm-lit-hdc")


def _to_windows_path(path):
    resolved = Path(path).resolve()
    parts = resolved.parts
    if len(parts) < 4 or parts[1] != "mnt":
        sys.exit(
            f"Windows hdc requires a staging path under /mnt/<drive>/..., got: {resolved}"
        )
    drive = parts[2].upper()
    tail = "\\".join(parts[3:])
    return f"{drive}:\\{tail}" if tail else f"{drive}:\\"


def _run_readelf(path):
    completed = subprocess.run(
        ["readelf", "-d", path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _read_elf_metadata(path):
    needed = []
    runpath_dirs = []
    seen_dirs = set()
    for line in _run_readelf(path).splitlines():
        needed_match = re.search(r"\(NEEDED\).*Shared library: \[(.*)\]", line)
        if needed_match:
            needed.append(needed_match.group(1))
            continue

        runpath_match = re.search(
            r"\((?:RUNPATH|RPATH)\).*Library runpath: \[(.*)\]", line
        )
        if not runpath_match:
            continue
        for entry in _split_path_list(runpath_match.group(1)):
            candidate = Path(entry)
            if not candidate.is_dir():
                continue
            key = str(candidate.resolve())
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            runpath_dirs.append(candidate)
    return needed, runpath_dirs


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
        library_dirs.append(parent)
    return library_dirs


def _find_dependency(name, search_dirs):
    for directory in search_dirs:
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _copy_needed_libraries(executables, execdir, staging_execdir):
    search_dirs = _collect_build_library_dirs(execdir)
    queue = [Path(exe).resolve() for exe in executables]
    visited = set()
    copied_names = {path.name for path in staging_execdir.glob("*") if path.is_file()}

    while queue:
        current = queue.pop(0)
        current_key = str(current)
        if current_key in visited:
            continue
        visited.add(current_key)

        needed, runpath_dirs = _read_elf_metadata(current)
        current_search_dirs = []
        current_seen = set()
        for directory in [*runpath_dirs, current.parent, *search_dirs]:
            resolved = Path(directory).resolve()
            key = str(resolved)
            if key in current_seen:
                continue
            current_seen.add(key)
            current_search_dirs.append(resolved)

        for directory in runpath_dirs:
            resolved = directory.resolve()
            if all(str(resolved) != str(existing) for existing in search_dirs):
                search_dirs.append(resolved)

        for lib_name in needed:
            dependency = _find_dependency(lib_name, current_search_dirs)
            if dependency is None:
                continue
            if lib_name not in copied_names:
                shutil.copy2(dependency, staging_execdir / lib_name)
                copied_names.add(lib_name)
            queue.append(dependency)


def _copy_execdir(execdir, staging_execdir):
    if staging_execdir.exists():
        shutil.rmtree(staging_execdir)
    shutil.copytree(execdir, staging_execdir)


def _iter_files(root):
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield Path(directory), filenames


def _is_test_executable(path):
    return path.endswith(".tmp.exe") and os.path.exists(path)


def _map_local_path(path, local_execdir, remote_execdir):
    normalized = os.path.normpath(path)
    execdir = os.path.normpath(local_execdir)
    if normalized == execdir:
        return remote_execdir
    prefix = execdir + os.sep
    if normalized.startswith(prefix):
        suffix = normalized[len(prefix) :].replace(os.sep, "/")
        return f"{remote_execdir}/{suffix}" if suffix else remote_execdir
    return path


def _map_value(value, local_execdir, remote_execdir):
    return _map_local_path(value, local_execdir, remote_execdir)


def _build_env_exports(env, prepend_env, local_execdir, remote_execdir, add_remote_libdir):
    commands = []
    mapped_env = {
        key: _map_value(value, local_execdir, remote_execdir) for key, value in env.items()
    }
    mapped_prepend = {
        key: _map_value(value, local_execdir, remote_execdir)
        for key, value in prepend_env.items()
    }

    if add_remote_libdir:
        existing = mapped_prepend.get("LD_LIBRARY_PATH")
        mapped_prepend["LD_LIBRARY_PATH"] = (
            f"{remote_execdir}:{existing}" if existing else remote_execdir
        )

    for key, value in mapped_env.items():
        commands.append(f"export {key}={shlex.quote(value)}")
    for key, value in mapped_prepend.items():
        commands.append(f"export {key}={shlex.quote(value)}${{{key}:+:{key}}}")
    return commands


def _hdc_prefix(hdc_path, server, target):
    prefix = [hdc_path]
    if server:
        prefix.extend(["-s", server])
    if target:
        prefix.extend(["-t", target])
    return prefix


def _run_hdc(
    prefix,
    args,
    *,
    timeout=None,
    capture_output=False,
    stdin=subprocess.DEVNULL,
    check=True,
    expect_stdout=None,
):
    kwargs = {
        "check": False,
        "timeout": timeout,
        "stdin": stdin,
        "text": True,
    }
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    completed = subprocess.run(prefix + args, **kwargs)
    output = completed.stdout if capture_output and completed.stdout else ""
    error_output = completed.stderr if capture_output and completed.stderr else ""
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"hdc command failed with exit code {completed.returncode}: "
            f"{shlex.join(prefix + args)}\n{output}{error_output}"
        )
    if check and expect_stdout is not None and expect_stdout not in output:
        raise RuntimeError(
            f"hdc command did not report success: {shlex.join(prefix + args)}\n{output}"
        )
    return completed


def _run_checked_shell(prefix, command, *, timeout=None):
    completed = _run_hdc(
        prefix,
        [
            "shell",
            f"{command}; rc=$?; printf '\\n{EXIT_MARKER}%d\\n' \"$rc\"",
        ],
        timeout=timeout,
        capture_output=True,
    )
    exit_code, output = _extract_exit_code(completed.stdout)
    if exit_code is None:
        raise RuntimeError(
            f"failed to parse shell exit code for: {command}\n{completed.stdout}"
        )
    if exit_code != 0:
        raise RuntimeError(
            f"remote shell command failed with exit code {exit_code}: {command}\n{output}"
        )
    return output


def _sync_tree_to_device(prefix, local_root, remote_root, use_windows_paths):
    _run_checked_shell(prefix, f"rm -rf {shlex.quote(remote_root)}")
    _run_checked_shell(prefix, f"mkdir -p {shlex.quote(remote_root)}")

    for directory, filenames in _iter_files(local_root):
        relative_dir = os.path.relpath(directory, local_root)
        device_dir = (
            remote_root
            if relative_dir == "."
            else f"{remote_root}/{relative_dir.replace(os.sep, '/')}"
        )
        _run_checked_shell(prefix, f"mkdir -p {shlex.quote(device_dir)}")
        for name in filenames:
            local_path = directory / name
            device_path = f"{device_dir}/{name}"
            source = _to_windows_path(local_path) if use_windows_paths else str(local_path)
            _run_hdc(
                prefix,
                ["file", "send", source, device_path],
                capture_output=True,
                expect_stdout="FileTransfer finish",
            )
            if name.endswith(".tmp.exe") or os.access(local_path, os.X_OK):
                _run_checked_shell(prefix, f"chmod +x {shlex.quote(device_path)}")


def _build_remote_command(command, local_execdir, remote_execdir, exports):
    mapped_command = [
        _map_local_path(arg, local_execdir, remote_execdir) for arg in command
    ]
    pieces = [f"cd {shlex.quote(remote_execdir)}"]
    pieces.extend(exports)
    pieces.append(shlex.join(mapped_command))
    pieces.append(f"rc=$?; printf '\\n{EXIT_MARKER}%d\\n' \"$rc\"")
    return " && ".join(pieces[:-2]) + f"; {pieces[-2]}; {pieces[-1]}"


def _extract_exit_code(output):
    match = re.search(rf"^(.*)\n{EXIT_MARKER}(\d+)\s*$", output, re.DOTALL)
    if not match:
        return None, output
    return int(match.group(2)), match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdc", default=os.environ.get("HDC", "hdc"))
    parser.add_argument("--hdc-server", default=os.environ.get("HDC_SERVER_IP_PORT"))
    parser.add_argument("--hdc-target", default=os.environ.get("HDC_UTID"))
    parser.add_argument("--execdir", required=True)
    parser.add_argument(
        "--remote-base-dir",
        default="/data/local/tmp/llvm-lit-hdc",
    )
    parser.add_argument(
        "--host-staging-root",
        default=os.environ.get("HDC_HOST_STAGING_ROOT"),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("HDC_TIMEOUT_SECONDS", "120")),
    )
    parser.add_argument("--env", nargs="*", default=[])
    parser.add_argument("--prepend_env", nargs="*", default=[])
    parser.add_argument("--keep-remote-dir", action="store_true")
    parser.add_argument("--keep-host-staging", action="store_true")
    parser.add_argument("command", nargs=argparse.ONE_OR_MORE)
    args = parser.parse_args()

    hdc_path = _resolve_hdc_binary(args.hdc)
    use_windows_paths = _is_windows_hdc(hdc_path)
    host_staging_root = (
        Path(args.host_staging_root)
        if args.host_staging_root
        else _default_host_staging_root(hdc_path)
    )
    prefix = _hdc_prefix(hdc_path, args.hdc_server, args.hdc_target)

    execdir = Path(args.execdir).resolve()
    run_id = uuid.uuid4().hex[:12]
    staging_execdir = host_staging_root / f"run-{run_id}" / execdir.name
    remote_execdir = f"{args.remote_base_dir}/run-{run_id}/{execdir.name}"

    env = _parse_kv(args.env)
    prepend_env = _parse_kv(args.prepend_env)

    executables = [arg for arg in args.command if _is_test_executable(arg)]
    host_staging_root.mkdir(parents=True, exist_ok=True)
    _copy_execdir(execdir, staging_execdir)
    _copy_needed_libraries(executables, execdir, staging_execdir)

    add_remote_libdir = any(staging_execdir.glob("*.so*"))
    exports = _build_env_exports(
        env,
        prepend_env,
        str(execdir),
        remote_execdir,
        add_remote_libdir,
    )

    try:
        _sync_tree_to_device(prefix, staging_execdir, remote_execdir, use_windows_paths)
        remote_command = _build_remote_command(
            args.command,
            str(execdir),
            remote_execdir,
            exports,
        )
        try:
            completed = _run_hdc(
                prefix,
                ["shell", remote_command],
                timeout=args.timeout_seconds,
                capture_output=True,
                stdin=None,
            )
        except subprocess.TimeoutExpired:
            print(
                f"hdc executor timed out after {args.timeout_seconds}s: {' '.join(args.command)}",
                file=sys.stderr,
            )
            return 124

        exit_code, output = _extract_exit_code(completed.stdout)
        if exit_code is None:
            print(
                "failed to parse exit code from hdc shell output",
                file=sys.stderr,
            )
            if completed.stdout:
                print(completed.stdout, end="", file=sys.stderr)
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            return 1

        if output:
            sys.stdout.write(output)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        return exit_code
    finally:
        if not args.keep_remote_dir:
            try:
                _run_checked_shell(prefix, f"rm -rf {shlex.quote(remote_execdir)}")
            except Exception:
                pass
        if not args.keep_host_staging:
            shutil.rmtree(staging_execdir.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
