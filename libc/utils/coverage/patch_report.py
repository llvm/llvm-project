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
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class DiffHunk:
    def __init__(self, header: str):
        self.header: str = header
        self.lines: List[Tuple[str, str, int]] = []  # (prefix, text, line_number)


class DiffParser:
    @staticmethod
    def parse(diff_source: str) -> Dict[str, List[DiffHunk]]:
        files: Dict[str, List[DiffHunk]] = {}
        current_file: Optional[str] = None
        current_hunk: Optional[DiffHunk] = None
        current_line_num: int = 0

        # Support both file path and raw diff string
        if os.path.isfile(diff_source):
            with open(diff_source, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = diff_source.splitlines(keepends=True)

        for line in lines:
            line = line.rstrip("\n")

            if line.startswith("+++ b/"):
                current_file = line[6:]
                files[current_file] = []
                current_hunk = None
                continue

            if line.startswith("+++ /dev/null"):
                current_file = None
                current_hunk = None
                continue

            if current_file is None:
                continue

            if line.startswith("@@"):
                match = re.search(r"\+([0-9]+)", line)
                if match:
                    current_line_num = int(match.group(1))
                    current_hunk = DiffHunk(line)
                    files[current_file].append(current_hunk)
                continue

            if current_hunk is None:
                continue

            if line.startswith("-"):
                continue
            elif line.startswith("+"):
                current_hunk.lines.append(("+", line[1:], current_line_num))
                current_line_num += 1
            elif line.startswith(" "):
                current_hunk.lines.append((" ", line[1:], current_line_num))
                current_line_num += 1

        return files


class CoverageJSONParser:
    @staticmethod
    def load(json_path: str) -> dict:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            sys.stderr.write(f"Error: Failed to parse coverage JSON: {e}\n")
            sys.exit(1)

    @staticmethod
    def extract_patch_matrix(
        cov_data: dict, diff_files: Dict[str, List[DiffHunk]]
    ) -> Dict[str, Dict[str, Any]]:
        coverage_matrix: Dict[str, Dict[str, Any]] = {
            fpath: {"covered": set(), "missed": set(), "mcdc_decisions": []}
            for fpath in diff_files.keys()
        }

        if "data" not in cov_data or not cov_data["data"]:
            return coverage_matrix

        for item in cov_data["data"][0].get("files", []):
            fpath = item.get("filename", "")
            rel_path = next(
                (rp for rp in diff_files.keys() if fpath.endswith(rp)), None
            )
            if not rel_path:
                continue

            # 1. Statement segments
            segments = item.get("segments", [])
            for i in range(len(segments) - 1):
                current = segments[i]
                nxt = segments[i + 1]

                line_start = current[0]
                line_end = nxt[0]
                count = current[2]
                has_count = current[3]

                if has_count:
                    for line_num in range(line_start, line_end + 1):
                        if count > 0:
                            coverage_matrix[rel_path]["covered"].add(line_num)
                        else:
                            coverage_matrix[rel_path]["missed"].add(line_num)

            # 2. MC/DC decision records (Clang 18+)
            mcdc_records = item.get("mcdc_records", [])
            for rec in mcdc_records:
                if len(rec) >= 10 and isinstance(rec[9], list):
                    l_start = rec[0]
                    l_end = rec[2]
                    conds = rec[9]
                    cov_conds = sum(1 for c in conds if c)
                    coverage_matrix[rel_path]["mcdc_decisions"].append(
                        {
                            "line_start": l_start,
                            "line_end": l_end,
                            "conditions": conds,
                            "covered": cov_conds,
                            "total": len(conds),
                        }
                    )

        return coverage_matrix


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
    coverage_matrix: Dict[str, Dict[str, Any]],
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

    total_mcdc_cov = 0
    total_mcdc_tot = 0
    total_decisions_count = 0
    fully_verified_decisions = 0
    file_mcdc_data: Dict[str, Tuple[int, int, List[str]]] = {}

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

        # Evaluate MC/DC decisions on modified lines
        f_mcdc_cov = 0
        f_mcdc_tot = 0
        f_dec_ver = 0
        f_dec_tot = 0
        condition_diagnostics = []
        unverified_decision_lines: Dict[int, List[str]] = {}

        for decision in data.get("mcdc_decisions", []):
            d_start = decision["line_start"]
            d_end = decision["line_end"]
            # Check if any modified line overlaps this decision range
            if any(d_start <= l <= d_end for l in added_lines):
                total_decisions_count += 1
                f_dec_tot += 1
                f_mcdc_cov += decision["covered"]
                f_mcdc_tot += decision["total"]
                if decision["covered"] == decision["total"]:
                    fully_verified_decisions += 1
                    f_dec_ver += 1
                    condition_diagnostics.append(
                        f"`L{d_start}`: {decision['covered']}/{decision['total']} verified"
                    )
                else:
                    uncovered_idx = [
                        f"C{i+1}"
                        for i, is_cov in enumerate(decision["conditions"])
                        if not is_cov
                    ]
                    unverified_str = ", ".join(uncovered_idx)
                    condition_diagnostics.append(
                        f"`L{d_start}`: {decision['covered']}/{decision['total']} verified ({unverified_str} unverified)"
                    )
                    for l in range(d_start, d_end + 1):
                        if l in added_lines:
                            unverified_decision_lines[l] = uncovered_idx

        if f_mcdc_tot > 0 or f_dec_tot > 0:
            total_mcdc_cov += f_mcdc_cov
            total_mcdc_tot += f_mcdc_tot
            file_mcdc_data[fpath] = (
                f_mcdc_cov,
                f_mcdc_tot,
                condition_diagnostics,
                f_dec_ver,
                f_dec_tot,
                unverified_decision_lines,
            )

    total_lines = total_covered + total_missed
    has_mcdc = total_mcdc_tot > 0

    if has_mcdc:
        print("## LLVM-libc MC/DC Patch Coverage Report\n")
    else:
        print("## LLVM-libc Patch Coverage Report\n")

    if total_lines == 0 or not active_files:
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
        return

    coverage_percent = (total_covered / total_lines) * 100
    mcdc_percent = (total_mcdc_cov / total_mcdc_tot * 100) if has_mcdc else 0.0

    if total_missed == 0:
        if not has_mcdc:
            print("> [!TIP]")
            print(f"> ### Patch Coverage: **{coverage_percent:.2f}%**")
            print(
                f"> All **{total_lines}** newly added or modified executable lines are covered."
            )
        elif total_mcdc_cov == total_mcdc_tot:
            print("> [!TIP]")
            print(
                f"> ### Patch Coverage: **{coverage_percent:.2f}% Line** | **100.00% MC/DC**"
            )
            print(
                f"> All **{total_lines}** executable lines and **{total_mcdc_tot}** boolean conditions across **{total_decisions_count}** decisions are covered."
            )
        else:
            print("> [!NOTE]")
            print(
                f"> ### Patch Coverage: **{coverage_percent:.2f}% Line** | **{mcdc_percent:.1f}% MC/DC**"
            )
            print(
                f"> Executed **{total_covered} / {total_lines}** lines. **{total_mcdc_cov} / {total_mcdc_tot}** boolean conditions achieved independence across **{fully_verified_decisions} / {total_decisions_count}** decisions."
            )
    else:
        print("> [!WARNING]")
        print(
            f"> ### Patch Coverage: **{coverage_percent:.2f}%** ({total_missed} Missed Lines)"
        )
        print(
            f"> **{total_missed}** unexecuted lines detected in patch."
        )
    print("")

    # Commit metadata and targets executed
    if base_sha and head_sha and base_branch and head_branch:
        print(
            f"- **Base Branch:** [`{base_branch}` ({base_sha[:7]})](https://github.com/{base_repo}/commit/{base_sha})"
        )
        print(
            f"- **Head Commit:** [`{head_branch}` ({head_sha[:7]})](https://github.com/{head_repo}/commit/{head_sha})"
        )
    if targets_str:
        targets_formatted = ", ".join(
            f"`{t.strip()}`" for t in targets_str.split() if t.strip()
        )
        print(f"- **Targeted Tests Executed:** {targets_formatted}")
    print("\n---\n")

    # Unified Coverage Breakdown Table
    print("### Coverage Breakdown")
    if has_mcdc:
        print(
            "| Modified Source File | Line Coverage | MC/DC Conditions | Decisions (Verified / Total) | Missed Lines | Unverified Conditions |"
        )
        print("| :--- | :---: | :---: | :---: | :---: | :--- |")
    else:
        print(
            "| Modified Source File | Patch Coverage | Covered / Total | Missed Lines | Unexecuted Line Spans |"
        )
        print("| :--- | :---: | :---: | :---: | :---: |")

    for fpath, (f_covered, f_missed, added_lines) in active_files.items():
        f_total = len(f_covered) + len(f_missed)
        f_pct = (len(f_covered) / f_total * 100) if f_total > 0 else 0.0
        line_spans = format_line_ranges(f_missed)
        file_link = f"[`{fpath}`](https://github.com/{head_repo}/blob/{head_sha or 'main'}/{fpath})"

        if has_mcdc:
            f_mc_data = file_mcdc_data.get(fpath, (0, 0, [], 0, 0, {}))
            f_mc_cov, f_mc_tot, diag_list, f_dec_ver, f_dec_tot = (
                f_mc_data[0],
                f_mc_data[1],
                f_mc_data[2],
                f_mc_data[3],
                f_mc_data[4],
            )
            if f_mc_tot > 0:
                f_mc_pct = f_mc_cov / f_mc_tot * 100
                mcdc_cell = f"**{f_mc_pct:.1f}%** ({f_mc_cov}/{f_mc_tot})"
            else:
                mcdc_cell = "N/A"

            dec_cell = (
                f"**{f_dec_ver} / {f_dec_tot}**" if f_dec_tot > 0 else "N/A"
            )
            diag_cell = "<br>".join(diag_list) if diag_list else "None"

            print(
                f"| {file_link} | **{f_pct:.2f}%** ({len(f_covered)}/{f_total}) | {mcdc_cell} | {dec_cell} | {len(f_missed)} | {diag_cell} |"
            )
        else:
            print(
                f"| {file_link} | **{f_pct:.2f}%** | {len(f_covered)} / {f_total} | {len(f_missed)} | {line_spans} |"
            )

    # Summary Row
    if has_mcdc:
        total_dec_cell = f"**{fully_verified_decisions} / {total_decisions_count}**"
        print(
            f"| **Total (Patch)** | **{coverage_percent:.2f}%** ({total_covered}/{total_lines}) | **{mcdc_percent:.1f}%** ({total_mcdc_cov}/{total_mcdc_tot}) | {total_dec_cell} | **{total_missed}** | - |"
        )
    else:
        print(
            f"| **Total (Patch)** | **{coverage_percent:.2f}%** | {total_covered} / {total_lines} | **{total_missed}** | - |"
        )
    print("")

    # Collapsible Source Map Diff
    print("<details>")
    print("<summary><b>View Annotated Patch Diff (Source Map)</b></summary>\n")

    for fpath, (f_covered, f_missed, added_lines) in active_files.items():
        hunks = diff_files.get(fpath, [])
        f_mc_data = file_mcdc_data.get(fpath, (0, 0, [], 0, 0, {}))
        unverified_lines = f_mc_data[5] if len(f_mc_data) > 5 else {}

        print(f"#### `{fpath}`")
        print("```diff")
        for hunk in hunks:
            print(hunk.header)
            for l_type, text, l_num in hunk.lines:
                if l_type == "+":
                    if l_num in f_missed:
                        print(f"- {text}  // [MISSED]")
                    elif l_num in unverified_lines:
                        unverified_conds = ", ".join(unverified_lines[l_num])
                        print(f"! {text}  // [PARTIAL MC/DC: {unverified_conds} unverified]")
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
    parser.add_argument(
        "targets", nargs="?", help="Space-separated list of executed test targets"
    )
    parser.add_argument(
        "base_repo",
        nargs="?",
        default="llvm/llvm-project",
        help="Base repository (e.g. llvm/llvm-project)",
    )
    parser.add_argument(
        "head_repo",
        nargs="?",
        default="llvm/llvm-project",
        help="Head repository (e.g. contributor/llvm-project)",
    )

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
