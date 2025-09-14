#!/usr/bin/env python3
#
# ===-----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#
# FIXME: Integrate with clang-tidy-diff.py


"""
Parallel clang-tidy runner
==========================

Runs clang-tidy over all files in a compilation database. Requires clang-tidy
and clang-apply-replacements in $PATH.

Example invocations.
- Run clang-tidy on all files in the current working directory with a default
  set of checks and show warnings in the cpp files and all project headers.
    run-clang-tidy.py $PWD

- Fix all header guards.
    run-clang-tidy.py -fix -checks=-*,llvm-header-guard

- Fix all header guards included from clang-tidy and header guards
  for clang-tidy headers.
    run-clang-tidy.py -fix -checks=-*,llvm-header-guard extra/clang-tidy \
                      -header-filter=extra/clang-tidy

Compilation database setup:
http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
"""

import argparse
import asyncio
from dataclasses import dataclass
import glob
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import ModuleType
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar


yaml: Optional[ModuleType] = None
try:
    import yaml
except ImportError:
    yaml = None


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to a bool following LLVM's CLI argument parsing."""

    val = val.lower()
    if val in ["", "true", "1"]:
        return True
    elif val in ["false", "0"]:
        return False

    # Return ArgumentTypeError so that argparse does not substitute its own error message
    raise argparse.ArgumentTypeError(
        f"'{val}' is invalid value for boolean argument! Try 0 or 1."
    )


def find_compilation_database(path: str) -> str:
    """Adjusts the directory until a compilation database is found."""
    result = os.path.realpath("./")
    while not os.path.isfile(os.path.join(result, path)):
        parent = os.path.dirname(result)
        if result == parent:
            print("Error: could not find compilation database.")
            sys.exit(1)
        result = parent
    return result


def get_tidy_invocation(
    f: Optional[str],
    clang_tidy_binary: str,
    checks: str,
    tmpdir: Optional[str],
    build_path: str,
    header_filter: Optional[str],
    allow_enabling_alpha_checkers: bool,
    extra_arg: List[str],
    extra_arg_before: List[str],
    quiet: bool,
    config_file_path: str,
    config: str,
    line_filter: Optional[str],
    use_color: bool,
    plugins: List[str],
    warnings_as_errors: Optional[str],
    exclude_header_filter: Optional[str],
    allow_no_checks: bool,
    store_check_profile: Optional[str],
) -> List[str]:
    """Gets a command line for clang-tidy."""
    start = [clang_tidy_binary]
    if allow_enabling_alpha_checkers:
        start.append("-allow-enabling-analyzer-alpha-checkers")
    if exclude_header_filter is not None:
        start.append(f"--exclude-header-filter={exclude_header_filter}")
    if header_filter is not None:
        start.append(f"-header-filter={header_filter}")
    if line_filter is not None:
        start.append(f"-line-filter={line_filter}")
    if use_color is not None:
        if use_color:
            start.append("--use-color")
        else:
            start.append("--use-color=false")
    if checks:
        start.append(f"-checks={checks}")
    if tmpdir is not None:
        start.append("-export-fixes")
        # Get a temporary file. We immediately close the handle so clang-tidy can
        # overwrite it.
        (handle, name) = tempfile.mkstemp(suffix=".yaml", dir=tmpdir)
        os.close(handle)
        start.append(name)
    for arg in extra_arg:
        start.append(f"-extra-arg={arg}")
    for arg in extra_arg_before:
        start.append(f"-extra-arg-before={arg}")
    start.append(f"-p={build_path}")
    if quiet:
        start.append("-quiet")
    if config_file_path:
        start.append(f"--config-file={config_file_path}")
    elif config:
        start.append(f"-config={config}")
    for plugin in plugins:
        start.append(f"-load={plugin}")
    if warnings_as_errors:
        start.append(f"--warnings-as-errors={warnings_as_errors}")
    if allow_no_checks:
        start.append("--allow-no-checks")
    if store_check_profile:
        start.append("--enable-check-profile")
        start.append(f"--store-check-profile={store_check_profile}")
    if f:
        start.append(f)
    return start


def merge_replacement_files(tmpdir: str, mergefile: str) -> None:
    """Merge all replacement files in a directory into a single file"""
    assert yaml
    # The fixes suggested by clang-tidy >= 4.0.0 are given under
    # the top level key 'Diagnostics' in the output yaml files
    mergekey = "Diagnostics"
    merged = []
    for replacefile in glob.iglob(os.path.join(tmpdir, "*.yaml")):
        content = yaml.safe_load(open(replacefile, "r"))
        if not content:
            continue  # Skip empty files.
        merged.extend(content.get(mergekey, []))

    if merged:
        # MainSourceFile: The key is required by the definition inside
        # include/clang/Tooling/ReplacementsYaml.h, but the value
        # is actually never used inside clang-apply-replacements,
        # so we set it to '' here.
        output = {"MainSourceFile": "", mergekey: merged}
        with open(mergefile, "w") as out:
            yaml.safe_dump(output, out)
    else:
        # Empty the file:
        open(mergefile, "w").close()


def aggregate_profiles(profile_dir: str) -> Dict[str, float]:
    """Aggregate timing data from multiple profile JSON files"""
    aggregated: Dict[str, float] = {}

    for profile_file in glob.iglob(os.path.join(profile_dir, "*.json")):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                profile_data: Dict[str, float] = data.get("profile", {})

                for key, value in profile_data.items():
                    if key.startswith("time.clang-tidy."):
                        if key in aggregated:
                            aggregated[key] += value
                        else:
                            aggregated[key] = value
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Error: invalid json file {profile_file}: {e}", file=sys.stderr)
            continue

    return aggregated


def print_profile_data(aggregated_data: Dict[str, float]) -> None:
    """Print aggregated checks profile data in the same format as clang-tidy"""
    if not aggregated_data:
        return

    # Extract checker names and their timing data
    checkers: Dict[str, Dict[str, float]] = {}
    for key, value in aggregated_data.items():
        parts = key.split(".")
        if len(parts) >= 4 and parts[0] == "time" and parts[1] == "clang-tidy":
            checker_name = ".".join(
                parts[2:-1]
            )  # Everything between "clang-tidy" and the timing type
            timing_type = parts[-1]  # wall, user, or sys

            if checker_name not in checkers:
                checkers[checker_name] = {"wall": 0.0, "user": 0.0, "sys": 0.0}

            checkers[checker_name][timing_type] = value

    if not checkers:
        return

    total_user = sum(data["user"] for data in checkers.values())
    total_sys = sum(data["sys"] for data in checkers.values())
    total_wall = sum(data["wall"] for data in checkers.values())

    sorted_checkers: List[Tuple[str, Dict[str, float]]] = sorted(
        checkers.items(), key=lambda x: x[1]["user"] + x[1]["sys"], reverse=True
    )

    def print_stderr(*args: Any, **kwargs: Any) -> None:
        print(*args, file=sys.stderr, **kwargs)

    print_stderr(
        "===-------------------------------------------------------------------------==="
    )
    print_stderr("                          clang-tidy checks profiling")
    print_stderr(
        "===-------------------------------------------------------------------------==="
    )
    print_stderr(
        f"  Total Execution Time: {total_user + total_sys:.4f} seconds ({total_wall:.4f} wall clock)\n"
    )

    # Calculate field widths based on the Total line which has the largest values
    total_combined = total_user + total_sys
    user_width = len(f"{total_user:.4f}")
    sys_width = len(f"{total_sys:.4f}")
    combined_width = len(f"{total_combined:.4f}")
    wall_width = len(f"{total_wall:.4f}")

    # Header with proper alignment
    additional_width = 9  # for " (100.0%)"
    user_header = "---User Time---".center(user_width + additional_width)
    sys_header = "--System Time--".center(sys_width + additional_width)
    combined_header = "--User+System--".center(combined_width + additional_width)
    wall_header = "---Wall Time---".center(wall_width + additional_width)

    print_stderr(
        f"   {user_header}   {sys_header}   {combined_header}   {wall_header}  --- Name ---"
    )

    for checker_name, data in sorted_checkers:
        user_time = data["user"]
        sys_time = data["sys"]
        wall_time = data["wall"]
        combined_time = user_time + sys_time

        user_percent = (user_time / total_user * 100) if total_user > 0 else 0
        sys_percent = (sys_time / total_sys * 100) if total_sys > 0 else 0
        combined_percent = (
            (combined_time / total_combined * 100) if total_combined > 0 else 0
        )
        wall_percent = (wall_time / total_wall * 100) if total_wall > 0 else 0

        user_str = f"{user_time:{user_width}.4f} ({user_percent:5.1f}%)"
        sys_str = f"{sys_time:{sys_width}.4f} ({sys_percent:5.1f}%)"
        combined_str = f"{combined_time:{combined_width}.4f} ({combined_percent:5.1f}%)"
        wall_str = f"{wall_time:{wall_width}.4f} ({wall_percent:5.1f}%)"

        print_stderr(
            f"   {user_str}   {sys_str}   {combined_str}   {wall_str}  {checker_name}"
        )

    user_total_str = f"{total_user:{user_width}.4f} (100.0%)"
    sys_total_str = f"{total_sys:{sys_width}.4f} (100.0%)"
    combined_total_str = f"{total_combined:{combined_width}.4f} (100.0%)"
    wall_total_str = f"{total_wall:{wall_width}.4f} (100.0%)"

    print_stderr(
        f"   {user_total_str}   {sys_total_str}   {combined_total_str}   {wall_total_str}  Total"
    )


def find_binary(arg: str, name: str, build_path: str) -> str:
    """Get the path for a binary or exit"""
    if arg:
        if shutil.which(arg):
            return arg
        else:
            raise SystemExit(
                f"error: passed binary '{arg}' was not found or is not executable"
            )

    built_path = os.path.join(build_path, "bin", name)
    binary = shutil.which(name) or shutil.which(built_path)
    if binary:
        return binary
    else:
        raise SystemExit(f"error: failed to find {name} in $PATH or at {built_path}")


def apply_fixes(
    args: argparse.Namespace, clang_apply_replacements_binary: str, tmpdir: str
) -> None:
    """Calls clang-apply-fixes on a given directory."""
    invocation = [clang_apply_replacements_binary]
    invocation.append("-ignore-insert-conflict")
    if args.format:
        invocation.append("-format")
    if args.style:
        invocation.append(f"-style={args.style}")
    invocation.append(tmpdir)
    subprocess.call(invocation)


# FIXME Python 3.12: This can be simplified out with run_with_semaphore[T](...).
T = TypeVar("T")


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    f: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    async with semaphore:
        return await f(*args, **kwargs)


@dataclass
class ClangTidyResult:
    filename: str
    invocation: List[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed: float


async def run_tidy(
    args: argparse.Namespace,
    name: str,
    clang_tidy_binary: str,
    tmpdir: str,
    build_path: str,
    store_check_profile: Optional[str],
) -> ClangTidyResult:
    """
    Runs clang-tidy on a single file and returns the result.
    """
    invocation = get_tidy_invocation(
        name,
        clang_tidy_binary,
        args.checks,
        tmpdir,
        build_path,
        args.header_filter,
        args.allow_enabling_alpha_checkers,
        args.extra_arg,
        args.extra_arg_before,
        args.quiet,
        args.config_file,
        args.config,
        args.line_filter,
        args.use_color,
        args.plugins,
        args.warnings_as_errors,
        args.exclude_header_filter,
        args.allow_no_checks,
        store_check_profile,
    )

    try:
        process = await asyncio.create_subprocess_exec(
            *invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        start = time.time()
        stdout, stderr = await process.communicate()
        end = time.time()
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        raise

    assert process.returncode is not None
    return ClangTidyResult(
        name,
        invocation,
        process.returncode,
        stdout.decode("UTF-8"),
        stderr.decode("UTF-8"),
        end - start,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runs clang-tidy over all files "
        "in a compilation database. Requires "
        "clang-tidy and clang-apply-replacements in "
        "$PATH or in your build directory."
    )
    parser.add_argument(
        "-allow-enabling-alpha-checkers",
        action="store_true",
        help="Allow alpha checkers from clang-analyzer.",
    )
    parser.add_argument(
        "-clang-tidy-binary", metavar="PATH", help="Path to clang-tidy binary."
    )
    parser.add_argument(
        "-clang-apply-replacements-binary",
        metavar="PATH",
        help="Path to clang-apply-replacements binary.",
    )
    parser.add_argument(
        "-checks",
        default=None,
        help="Checks filter, when not specified, use clang-tidy default.",
    )
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "-config",
        default=None,
        help="Specifies a configuration in YAML/JSON format: "
        "  -config=\"{Checks: '*', "
        '                       CheckOptions: {x: y}}" '
        "When the value is empty, clang-tidy will "
        "attempt to find a file named .clang-tidy for "
        "each source file in its parent directories.",
    )
    config_group.add_argument(
        "-config-file",
        default=None,
        help="Specify the path of .clang-tidy or custom config "
        "file: e.g. -config-file=/some/path/myTidyConfigFile. "
        "This option internally works exactly the same way as "
        "-config option after reading specified config file. "
        "Use either -config-file or -config, not both.",
    )
    parser.add_argument(
        "-exclude-header-filter",
        default=None,
        help="Regular expression matching the names of the "
        "headers to exclude diagnostics from. Diagnostics from "
        "the main file of each translation unit are always "
        "displayed.",
    )
    parser.add_argument(
        "-header-filter",
        default=None,
        help="Regular expression matching the names of the "
        "headers to output diagnostics from. Diagnostics from "
        "the main file of each translation unit are always "
        "displayed.",
    )
    parser.add_argument(
        "-source-filter",
        default=None,
        help="Regular expression matching the names of the "
        "source files from compilation database to output "
        "diagnostics from.",
    )
    parser.add_argument(
        "-line-filter",
        default=None,
        help="List of files and line ranges to output diagnostics from.",
    )
    if yaml:
        parser.add_argument(
            "-export-fixes",
            metavar="file_or_directory",
            dest="export_fixes",
            help="A directory or a yaml file to store suggested fixes in, "
            "which can be applied with clang-apply-replacements. If the "
            "parameter is a directory, the fixes of each compilation unit are "
            "stored in individual yaml files in the directory.",
        )
    else:
        parser.add_argument(
            "-export-fixes",
            metavar="directory",
            dest="export_fixes",
            help="A directory to store suggested fixes in, which can be applied "
            "with clang-apply-replacements. The fixes of each compilation unit are "
            "stored in individual yaml files in the directory.",
        )
    parser.add_argument(
        "-j",
        type=int,
        default=0,
        help="Number of tidy instances to be run in parallel.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=[".*"],
        help="Files to be processed (regex on path).",
    )
    parser.add_argument("-fix", action="store_true", help="apply fix-its.")
    parser.add_argument(
        "-format", action="store_true", help="Reformat code after applying fixes."
    )
    parser.add_argument(
        "-style",
        default="file",
        help="The style of reformat code after applying fixes.",
    )
    parser.add_argument(
        "-use-color",
        type=strtobool,
        nargs="?",
        const=True,
        help="Use colors in diagnostics, overriding clang-tidy's"
        " default behavior. This option overrides the 'UseColor"
        "' option in .clang-tidy file, if any.",
    )
    parser.add_argument(
        "-p", dest="build_path", help="Path used to read a compile command database."
    )
    parser.add_argument(
        "-extra-arg",
        dest="extra_arg",
        action="append",
        default=[],
        help="Additional argument to append to the compiler command line.",
    )
    parser.add_argument(
        "-extra-arg-before",
        dest="extra_arg_before",
        action="append",
        default=[],
        help="Additional argument to prepend to the compiler command line.",
    )
    parser.add_argument(
        "-quiet", action="store_true", help="Run clang-tidy in quiet mode."
    )
    parser.add_argument(
        "-load",
        dest="plugins",
        action="append",
        default=[],
        help="Load the specified plugin in clang-tidy.",
    )
    parser.add_argument(
        "-warnings-as-errors",
        default=None,
        help="Upgrades warnings to errors. Same format as '-checks'.",
    )
    parser.add_argument(
        "-allow-no-checks",
        action="store_true",
        help="Allow empty enabled checks.",
    )
    parser.add_argument(
        "-enable-check-profile",
        action="store_true",
        help="Enable per-check timing profiles, and print a report",
    )
    parser.add_argument(
        "-hide-progress",
        action="store_true",
        help="Hide progress",
    )
    args = parser.parse_args()

    db_path = "compile_commands.json"

    if args.build_path is not None:
        build_path = args.build_path
    else:
        # Find our database
        build_path = find_compilation_database(db_path)

    clang_tidy_binary = find_binary(args.clang_tidy_binary, "clang-tidy", build_path)

    if args.fix:
        clang_apply_replacements_binary = find_binary(
            args.clang_apply_replacements_binary, "clang-apply-replacements", build_path
        )

    combine_fixes = False
    export_fixes_dir: Optional[str] = None
    delete_fixes_dir = False
    if args.export_fixes is not None:
        # if a directory is given, create it if it does not exist
        if args.export_fixes.endswith(os.path.sep) and not os.path.isdir(
            args.export_fixes
        ):
            os.makedirs(args.export_fixes)

        if not os.path.isdir(args.export_fixes):
            if not yaml:
                raise RuntimeError(
                    "Cannot combine fixes in one yaml file. Either install PyYAML or specify an output directory."
                )

            combine_fixes = True

        if os.path.isdir(args.export_fixes):
            export_fixes_dir = args.export_fixes

    if export_fixes_dir is None and (args.fix or combine_fixes):
        export_fixes_dir = tempfile.mkdtemp()
        delete_fixes_dir = True

    profile_dir: Optional[str] = None
    if args.enable_check_profile:
        profile_dir = tempfile.mkdtemp()

    try:
        invocation = get_tidy_invocation(
            None,
            clang_tidy_binary,
            args.checks,
            None,
            build_path,
            args.header_filter,
            args.allow_enabling_alpha_checkers,
            args.extra_arg,
            args.extra_arg_before,
            args.quiet,
            args.config_file,
            args.config,
            args.line_filter,
            args.use_color,
            args.plugins,
            args.warnings_as_errors,
            args.exclude_header_filter,
            args.allow_no_checks,
            None,  # No profiling for the list-checks invocation
        )
        invocation.append("-list-checks")
        invocation.append("-")
        # Even with -quiet we still want to check if we can call clang-tidy.
        subprocess.check_call(
            invocation, stdout=subprocess.DEVNULL if args.quiet else None
        )
    except:
        print("Unable to run clang-tidy.", file=sys.stderr)
        sys.exit(1)

    # Load the database and extract all files.
    with open(os.path.join(build_path, db_path)) as f:
        database = json.load(f)
    files = {os.path.abspath(os.path.join(e["directory"], e["file"])) for e in database}
    number_files_in_database = len(files)

    # Filter source files from compilation database.
    if args.source_filter:
        try:
            source_filter_re = re.compile(args.source_filter)
        except:
            print(
                "Error: unable to compile regex from arg -source-filter:",
                file=sys.stderr,
            )
            traceback.print_exc()
            sys.exit(1)
        files = {f for f in files if source_filter_re.match(f)}

    max_task = args.j
    if max_task == 0:
        max_task = multiprocessing.cpu_count()

    # Build up a big regexy filter from all command line arguments.
    file_name_re = re.compile("|".join(args.files))
    files = {f for f in files if file_name_re.search(f)}

    if not args.hide_progress:
        print(
            f"Running clang-tidy in {max_task} threads for {len(files)} files "
            f"out of {number_files_in_database} in compilation database ..."
        )

    returncode = 0
    semaphore = asyncio.Semaphore(max_task)
    tasks = [
        asyncio.create_task(
            run_with_semaphore(
                semaphore,
                run_tidy,
                args,
                f,
                clang_tidy_binary,
                export_fixes_dir,
                build_path,
                profile_dir,
            )
        )
        for f in files
    ]

    try:
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            if result.returncode != 0:
                returncode = 1
                if result.returncode < 0:
                    result.stderr += f"{result.filename}: terminated by signal {-result.returncode}\n"
            progress = f"[{i + 1: >{len(f'{len(files)}')}}/{len(files)}]"
            runtime = f"[{result.elapsed:.1f}s]"
            if not args.hide_progress:
                print(f"{progress}{runtime} {' '.join(result.invocation)}")
            if result.stdout:
                print(result.stdout, end=("" if result.stderr else "\n"))
            if result.stderr:
                print(result.stderr)
    except asyncio.CancelledError:
        if not args.hide_progress:
            print("\nCtrl-C detected, goodbye.")
        for task in tasks:
            task.cancel()
        if delete_fixes_dir:
            assert export_fixes_dir
            shutil.rmtree(export_fixes_dir)
        if profile_dir:
            shutil.rmtree(profile_dir)
        return

    if args.enable_check_profile and profile_dir:
        # Ensure all clang-tidy stdout is flushed before printing profiling
        sys.stdout.flush()
        aggregated_data = aggregate_profiles(profile_dir)
        if aggregated_data:
            print_profile_data(aggregated_data)
        else:
            print("No profiling data found.")

    if combine_fixes:
        if not args.hide_progress:
            print(f"Writing fixes to {args.export_fixes} ...")
        try:
            assert export_fixes_dir
            merge_replacement_files(export_fixes_dir, args.export_fixes)
        except:
            print("Error exporting fixes.\n", file=sys.stderr)
            traceback.print_exc()
            returncode = 1

    if args.fix:
        if not args.hide_progress:
            print("Applying fixes ...")
        try:
            assert export_fixes_dir
            apply_fixes(args, clang_apply_replacements_binary, export_fixes_dir)
        except:
            print("Error applying fixes.\n", file=sys.stderr)
            traceback.print_exc()
            returncode = 1

    if delete_fixes_dir:
        assert export_fixes_dir
        shutil.rmtree(export_fixes_dir)
    if profile_dir:
        shutil.rmtree(profile_dir)
    sys.exit(returncode)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
