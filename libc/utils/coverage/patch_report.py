#!/usr/bin/env python3
#
# ====- Generate patch coverage reports ------------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure local module import works regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from parser import CoverageJSONParser, DiffHunk, DiffParser

def is_executable_line(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    # Comments
    if (
        s.startswith("//")
        or s.startswith("/*")
        or s.startswith("*")
        or s.startswith("*/")
    ):
        return False
    # Structural braces and colons
    if s in ("{", "}", "};", "{};") or s.startswith(":"):
        return False
    # Preprocessor directives
    if s.startswith("#"):
        return False
    # Declarations / keywords / attributes
    if (
        s.startswith("namespace ")
        or s.startswith("extern ")
        or s.startswith("using ")
        or s.startswith("__attribute__")
        or s.startswith("template")
        or s.startswith("typedef ")
        or s.startswith("struct ")
        or s.startswith("class ")
        or s.startswith("enum ")
    ):
        return False
    return True


def format_line_ranges(lines: Set[int]) -> str:
    if not lines:
        return "None"
    sorted_lines = sorted(lines)
    ranges = []
    start = sorted_lines[0]
    end = sorted_lines[0]
    for n in sorted_lines[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"`L{start}-L{end}`" if start != end else f"`L{start}`")
            start = end = n
    ranges.append(f"`L{start}-L{end}`" if start != end else f"`L{start}`")
    return ", ".join(ranges)


def render_patch_report(
    diff_files: Dict[str, List[DiffHunk]],
    coverage_matrix: Dict[str, Dict[str, Set[int]]],
    base_sha: Optional[str],
    head_sha: Optional[str],
    base_branch: Optional[str],
    head_branch: Optional[str],
    targets_str: Optional[str] = None,
    base_repo: str = "llvm/llvm-project",
    head_repo: str = "llvm/llvm-project",
) -> None:
    total_covered = 0
    total_missed = 0
    active_files = {}

    for fpath, data in coverage_matrix.items():
        added_lines: Set[int] = set()
        for hunk in diff_files.get(fpath, []):
            for l_type, text, l_num in hunk.lines:
                if l_type == "+":
                    if not is_executable_line(text):
                        continue
                    added_lines.add(l_num)

        if not added_lines:
            continue

        f_covered = added_lines.intersection(data["covered"])
        f_missed = (added_lines.intersection(data["missed"])) - f_covered

        if len(data["covered"]) > 0 or len(data["missed"]) > 0:
            total_covered += len(f_covered)
            total_missed += len(f_missed)
            active_files[fpath] = (f_covered, f_missed, added_lines)
        else:
            total_missed += len(added_lines)
            active_files[fpath] = (set(), added_lines, added_lines)

    total_lines = total_covered + total_missed

    if total_lines == 0 or not active_files:
        print("## LLVM-libc Patch Coverage Report\n")
        if base_sha and head_sha and base_branch and head_branch:
            print(
                f"- **Base Branch:** [`{base_branch}` ({base_sha[:7]})](https://github.com/{base_repo}/commit/{base_sha})"
            )
            print(
                f"- **Head Commit:** [`{head_branch}` ({head_sha[:7]})](https://github.com/{head_repo}/commit/{head_sha})\n"
            )
            print("---\n")
        print("> [!NOTE]")
        print("> ### Coverage Validated")
        print("> No `.cpp` source files in `libc/src/` were modified in this patch.")
        sys.exit(0)

    print("## LLVM-libc Patch Coverage Report\n")

    coverage_percent = (total_covered / total_lines) * 100

    # Modern GitHub UI Alert Card
    if total_missed == 0:
        print("> [!TIP]")
        print(f"> ### Patch Coverage: **{coverage_percent:.2f}%** (PASSED)")
        print(
            f"> All **{total_lines}** newly added or modified executable lines are covered by targeted unit tests."
        )
    else:
        print("> [!WARNING]")
        print(f"> ### Patch Coverage: **{coverage_percent:.2f}%** ({total_missed} Missed Lines)")
        print(
            f"> **{total_missed}** unexecuted line(s) detected in your patch. Please review the missing lines below."
        )
    print("")

    # Commit metadata and targets executed with exact upstream and fork repositories
    if base_sha and head_sha and base_branch and head_branch:
        print(
            f"- **Base Branch:** [`{base_branch}` ({base_sha[:7]})](https://github.com/{base_repo}/commit/{base_sha})"
        )
        print(
            f"- **Head Commit:** [`{head_branch}` ({head_sha[:7]})](https://github.com/{head_repo}/commit/{head_sha})"
        )
    if targets_str:
        targets_formatted = ", ".join(f"`{t.strip()}`" for t in targets_str.split() if t.strip())
        print(f"- **Targeted Tests Executed:** {targets_formatted}")
    print("\n---\n")

    # Executive Summary Table
    status_label = "**PASSED**" if total_missed == 0 else "**ACTION REQUIRED**"
    commit_link = f"[`{head_sha[:7]}`](https://github.com/{head_repo}/commit/{head_sha})" if head_sha else "HEAD"

    print("### Executive Summary")
    print(f"The code coverage on the recent commit {commit_link} is **{coverage_percent:.2f}%**.")
    print("")
    print("| Metric | Value | Status |")
    print("| :--- | :---: | :---: |")
    print(f"| **Patch Line Coverage** | **{coverage_percent:.2f}%** | {status_label} |")
    print(f"| **Executable Lines Evaluated** | **{total_lines}** | — |")
    print(f"| **Covered Lines** | **{total_covered}** | {coverage_percent:.1f}% |")
    print(f"| **Unexecuted Lines** | **{total_missed}** | {'0' if total_missed == 0 else str(total_missed)} |")
    print("")

    # Modified Files Impact Table
    print("### Modified Files Impact")
    print("| Modified Source File | Patch Coverage | Covered / Total | Missed Lines | Unexecuted Line Spans |")
    print("| :--- | :---: | :---: | :---: | :---: |")

    for fpath, (f_covered, f_missed, added_lines) in active_files.items():
        f_total = len(f_covered) + len(f_missed)
        f_pct = (len(f_covered) / f_total * 100) if f_total > 0 else 0.0
        line_spans = format_line_ranges(f_missed)
        file_link = f"[`{fpath}`](https://github.com/{head_repo}/blob/{head_sha or 'main'}/{fpath})"
        print(
            f"| {file_link} | **{f_pct:.2f}%** | {len(f_covered)} / {f_total} | {len(f_missed)} | {line_spans} |"
        )
    print("")

    # Collapsible Source Map Diff
    print("<details>")
    print("<summary><b>View Annotated Patch Diff (Source Map)</b></summary>\n")

    for fpath, (f_covered, f_missed, added_lines) in active_files.items():
        hunks = diff_files.get(fpath, [])
        print(f"#### `{fpath}`")
        print("```diff")
        for hunk in hunks:
            print(hunk.header)
            for l_type, text, l_num in hunk.lines:
                if l_type == "+":
                    if l_num in f_missed:
                        print(f"- {text}")
                    elif l_num in f_covered:
                        print(f"+ {text}")
                    else:
                        print(f"  {text}")
                elif l_type == " ":
                    print(f"  {text}")
        print("```\n")
    print("</details>")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLVM-libc Patch Coverage Analyzer")
    parser.add_argument("diff_file", help="Path to unified diff file")
    parser.add_argument("json_file", help="Path to llvm-cov export JSON file")
    parser.add_argument("base_sha", nargs="?", help="Base commit SHA")
    parser.add_argument("head_sha", nargs="?", help="Head commit SHA")
    parser.add_argument("base_branch", nargs="?", help="Base branch name")
    parser.add_argument("head_branch", nargs="?", help="Head branch name")
    parser.add_argument("targets", nargs="?", help="Space-separated list of executed test targets")
    parser.add_argument("base_repo", nargs="?", default="llvm/llvm-project", help="Base repository (e.g. llvm/llvm-project)")
    parser.add_argument("head_repo", nargs="?", default="llvm/llvm-project", help="Head repository (e.g. tapiwagonga/llvm-project)")

    args = parser.parse_args()

    diff_files = DiffParser.parse(args.diff_file)
    cov_data = CoverageJSONParser.load(args.json_file)
    coverage_matrix = CoverageJSONParser.extract_patch_matrix(cov_data, diff_files)

    render_patch_report(
        diff_files,
        coverage_matrix,
        args.base_sha,
        args.head_sha,
        args.base_branch,
        args.head_branch,
        args.targets,
        args.base_repo,
        args.head_repo,
    )


if __name__ == "__main__":
    main()
