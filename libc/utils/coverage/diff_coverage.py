#!/usr/bin/env python3
#
# ====- Generate diff coverage reports -------------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

"""
Standalone analyzer for evaluating diff-level statement, branch, and MC/DC coverage.

This script parses unified git diffs alongside `llvm-cov export` JSON summaries,
correlates added/modified lines with execution counts and boolean decision records,
and outputs formatted Markdown reports for CI job summaries and PR comments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

DEFAULT_BASE_REPO = "llvm/llvm-project"
DEFAULT_HEAD_REPO = "llvm/llvm-project"

COMMENT_PREFIXES = ("//", "/*", "*", "*/")
STRUCTURAL_TOKENS = ("{", "}", "};", "{};")
DECLARATION_PREFIXES = (
    "namespace ",
    "extern ",
    "using ",
    "__attribute__",
    "template",
    "typedef ",
)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class DiffHunk:
    """Represents a unified diff hunk with its header and line tokens."""

    header: str
    lines: List[Tuple[str, str, int]] = field(default_factory=list)  # (prefix, text, line_number)


@dataclass
class FilePatchMetrics:
    """Encapsulates coverage metrics and decision records for a single modified file."""

    fpath: str
    covered_lines: Set[int] = field(default_factory=set)
    missed_lines: Set[int] = field(default_factory=set)
    added_lines: Set[int] = field(default_factory=set)
    mcdc_cov: int = 0
    mcdc_tot: int = 0
    decisions_ver: int = 0
    decisions_tot: int = 0
    condition_diagnostics: List[str] = field(default_factory=list)
    unverified_decision_lines: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def total_lines(self) -> int:
        """Total executable lines evaluated in this file."""
        return len(self.covered_lines) + len(self.missed_lines)

    @property
    def line_coverage_pct(self) -> float:
        """Percentage of executed patch lines."""
        return (len(self.covered_lines) / self.total_lines * 100.0) if self.total_lines > 0 else 0.0

    @property
    def mcdc_coverage_pct(self) -> float:
        """Percentage of independent boolean conditions evaluated."""
        return (self.mcdc_cov / self.mcdc_tot * 100.0) if self.mcdc_tot > 0 else 0.0


@dataclass
class PatchCoverageSummary:
    """Aggregated coverage statistics across all modified files in the patch."""

    files: Dict[str, FilePatchMetrics] = field(default_factory=dict)
    total_covered_lines: int = 0
    total_missed_lines: int = 0
    total_mcdc_cov: int = 0
    total_mcdc_tot: int = 0
    total_decisions_count: int = 0
    fully_verified_decisions: int = 0

    @property
    def total_lines(self) -> int:
        """Total executable lines across all modified files in the patch."""
        return self.total_covered_lines + self.total_missed_lines

    @property
    def line_coverage_pct(self) -> float:
        """Aggregated patch line coverage percentage."""
        return (self.total_covered_lines / self.total_lines * 100.0) if self.total_lines > 0 else 0.0

    @property
    def mcdc_coverage_pct(self) -> float:
        """Aggregated patch MC/DC condition coverage percentage."""
        return (self.total_mcdc_cov / self.total_mcdc_tot * 100.0) if self.total_mcdc_tot > 0 else 0.0

    @property
    def has_mcdc(self) -> bool:
        """Returns True if any MC/DC decision records intersect the patch."""
        return self.total_mcdc_tot > 0


# -----------------------------------------------------------------------------
# Parsing Utilities
# -----------------------------------------------------------------------------

class DiffParser:
    """Parses unified diff outputs into structured file hunks with line numbers."""

    @staticmethod
    def parse(diff_source: str) -> Dict[str, List[DiffHunk]]:
        """Parses a diff file path or raw diff string into a mapping of file path to hunks."""
        files: Dict[str, List[DiffHunk]] = {}
        current_file: Optional[str] = None
        current_hunk: Optional[DiffHunk] = None
        current_line_num: int = 0

        if os.path.isfile(diff_source):
            with open(diff_source, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = diff_source.splitlines(keepends=True)

        for raw_line in lines:
            line = raw_line.rstrip("\n")

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
                    current_hunk = DiffHunk(header=line)
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
    """Parses and extracts statement segments and MC/DC records from llvm-cov JSON export."""

    @staticmethod
    def load(json_path: str) -> dict:
        """Loads JSON file from disk with error reporting."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            sys.stderr.write(f"Error: Failed to parse coverage JSON from '{json_path}': {err}\n")
            sys.exit(1)

    @staticmethod
    def extract_patch_matrix(
        cov_data: dict, diff_files: Dict[str, List[DiffHunk]]
    ) -> Dict[str, Dict[str, Any]]:
        """Maps statement coverage segments and MC/DC decision records to modified files."""
        coverage_matrix: Dict[str, Dict[str, Any]] = {
            fpath: {"covered": set(), "missed": set(), "mcdc_decisions": []}
            for fpath in diff_files.keys()
        }

        if "data" not in cov_data or not cov_data["data"]:
            return coverage_matrix

        for item in cov_data["data"][0].get("files", []):
            fpath = item.get("filename", "")
            rel_path = next(
                (
                    rp
                    for rp in diff_files.keys()
                    if fpath == rp or fpath.endswith("/" + rp) or rp.endswith("/" + fpath)
                ),
                None,
            )
            if not rel_path:
                continue

            # 1. Process statement coverage segments
            segments = item.get("segments", [])
            for i, current in enumerate(segments):
                line_start = current[0]
                count = current[2]
                has_count = current[3]

                if not has_count:
                    continue

                if i < len(segments) - 1:
                    nxt = segments[i + 1]
                    line_end = nxt[0]
                    end_range = line_end if line_end > line_start else line_start + 1
                else:
                    end_range = line_start + 1

                for line_num in range(line_start, end_range):
                    if count > 0:
                        coverage_matrix[rel_path]["covered"].add(line_num)
                    else:
                        coverage_matrix[rel_path]["missed"].add(line_num)

            # 2. Process MC/DC decision records
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
    """Filters out non-executable code lines (comments, braces, pure declarations)."""
    s = text.strip()
    if not s:
        return False
    if any(s.startswith(prefix) for prefix in COMMENT_PREFIXES):
        return False
    if s in STRUCTURAL_TOKENS or s.startswith(":"):
        return False
    if s.startswith("#"):
        return False
    if any(s.startswith(prefix) for prefix in DECLARATION_PREFIXES):
        return False
    if s.startswith("struct ") or s.startswith("class ") or s.startswith("enum "):
        if "{" in s or (s.endswith(";") and "=" not in s and "(" not in s):
            return False
    return True


def format_line_ranges(lines: Set[int]) -> str:
    """Formats an integer set of line numbers into concise span representations."""
    if not lines:
        return "None"
    sorted_lines = sorted(lines)
    ranges: List[str] = []
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


# -----------------------------------------------------------------------------
# Statistics Calculation
# -----------------------------------------------------------------------------

def calculate_patch_statistics(
    diff_files: Dict[str, List[DiffHunk]],
    coverage_matrix: Dict[str, Dict[str, Any]],
) -> PatchCoverageSummary:
    """Calculates granular line, branch, and MC/DC statistics for all modified patch files."""
    summary = PatchCoverageSummary()

    for fpath, data in coverage_matrix.items():
        added_lines: Set[int] = set()
        for hunk in diff_files.get(fpath, []):
            for l_type, text, l_num in hunk.lines:
                if l_type == "+" and is_executable_line(text):
                    added_lines.add(l_num)

        if not added_lines:
            continue

        f_covered = added_lines.intersection(data["covered"])
        f_missed = (added_lines.intersection(data["missed"])) - f_covered

        file_metric = FilePatchMetrics(
            fpath=fpath,
            added_lines=added_lines,
        )

        if len(data["covered"]) > 0 or len(data["missed"]) > 0:
            file_metric.covered_lines = f_covered
            file_metric.missed_lines = f_missed
            summary.total_covered_lines += len(f_covered)
            summary.total_missed_lines += len(f_missed)
        else:
            file_metric.missed_lines = added_lines
            summary.total_missed_lines += len(added_lines)

        # Evaluate MC/DC decision records intersecting modified lines
        for decision in data.get("mcdc_decisions", []):
            d_start = decision["line_start"]
            d_end = decision["line_end"]
            if any(d_start <= l <= d_end for l in added_lines):
                summary.total_decisions_count += 1
                file_metric.decisions_tot += 1
                file_metric.mcdc_cov += decision["covered"]
                file_metric.mcdc_tot += decision["total"]
                summary.total_mcdc_cov += decision["covered"]
                summary.total_mcdc_tot += decision["total"]

                if decision["covered"] == decision["total"]:
                    summary.fully_verified_decisions += 1
                    file_metric.decisions_ver += 1
                    file_metric.condition_diagnostics.append(
                        f"`L{d_start}`: {decision['covered']}/{decision['total']} verified"
                    )
                else:
                    uncovered_idx = [
                        f"C{i+1}"
                        for i, is_cov in enumerate(decision["conditions"])
                        if not is_cov
                    ]
                    unverified_str = ", ".join(uncovered_idx)
                    file_metric.condition_diagnostics.append(
                        f"`L{d_start}`: {decision['covered']}/{decision['total']} verified ({unverified_str} unverified)"
                    )
                    for l in range(d_start, d_end + 1):
                        if l in added_lines:
                            file_metric.unverified_decision_lines[l] = uncovered_idx

        summary.files[fpath] = file_metric

    return summary


# -----------------------------------------------------------------------------
# Report Formatting
# -----------------------------------------------------------------------------

def format_status_banner(summary: PatchCoverageSummary) -> str:
    """Generates the executive summary callout block."""
    lines: List[str] = []
    if summary.total_missed_lines == 0:
        if not summary.has_mcdc:
            lines.append("> [!TIP]")
            lines.append(f"> ### Patch Coverage: **{summary.line_coverage_pct:.2f}%**")
            lines.append(
                f"> All **{summary.total_lines}** newly added or modified executable lines are covered."
            )
        elif summary.total_mcdc_cov == summary.total_mcdc_tot:
            lines.append("> [!TIP]")
            lines.append(
                f"> ### Patch Coverage: **{summary.line_coverage_pct:.2f}% Line** | **100.00% MC/DC**"
            )
            lines.append(
                f"> All **{summary.total_lines}** executable lines and **{summary.total_mcdc_tot}** boolean conditions across **{summary.total_decisions_count}** decisions are covered."
            )
        else:
            lines.append("> [!NOTE]")
            lines.append(
                f"> ### Patch Coverage: **{summary.line_coverage_pct:.2f}% Line** | **{summary.mcdc_coverage_pct:.1f}% MC/DC**"
            )
            lines.append(
                f"> Executed **{summary.total_covered_lines} / {summary.total_lines}** lines. **{summary.total_mcdc_cov} / {summary.total_mcdc_tot}** boolean conditions achieved independence across **{summary.fully_verified_decisions} / {summary.total_decisions_count}** decisions."
            )
    else:
        lines.append("> [!WARNING]")
        lines.append(
            f"> ### Patch Coverage: **{summary.line_coverage_pct:.2f}%** ({summary.total_missed_lines} Missed Lines)"
        )
        lines.append(
            f"> **{summary.total_missed_lines}** unexecuted lines detected in patch."
        )
    return "\n".join(lines)


def format_metadata_section(
    base_sha: Optional[str],
    head_sha: Optional[str],
    base_branch: Optional[str],
    head_branch: Optional[str],
    targets_str: Optional[str] = None,
    base_repo: str = DEFAULT_BASE_REPO,
    head_repo: str = DEFAULT_HEAD_REPO,
) -> str:
    """Formats Git commit and target test metadata."""
    lines: List[str] = []
    if base_sha and head_sha and base_branch and head_branch:
        lines.append(
            f"- **Base Branch:** [`{base_branch}` ({base_sha[:7]})](https://github.com/{base_repo}/commit/{base_sha})"
        )
        lines.append(
            f"- **Head Commit:** [`{head_branch}` ({head_sha[:7]})](https://github.com/{head_repo}/commit/{head_sha})"
        )
    if targets_str:
        targets_formatted = ", ".join(
            f"`{t.strip()}`" for t in targets_str.split() if t.strip()
        )
        lines.append(f"- **Targeted Tests Executed:** {targets_formatted}")
    return "\n".join(lines)


def format_breakdown_table(
    summary: PatchCoverageSummary,
    head_repo: str = DEFAULT_HEAD_REPO,
    head_sha: Optional[str] = None,
) -> str:
    """Generates the Markdown table breaking down coverage per source file."""
    lines: List[str] = ["### Coverage Breakdown"]
    if summary.has_mcdc:
        lines.append(
            "| Modified Source File | Line Coverage | MC/DC Conditions | Decisions (Verified / Total) | Missed Lines | Unverified Conditions |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    else:
        lines.append(
            "| Modified Source File | Patch Coverage | Covered / Total | Missed Lines | Unexecuted Line Spans |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for fpath, file_metric in summary.files.items():
        file_link = f"[`{fpath}`](https://github.com/{head_repo}/blob/{head_sha or 'main'}/{fpath})"
        f_missed = file_metric.missed_lines
        f_cov_len = len(file_metric.covered_lines)
        f_tot_len = file_metric.total_lines

        if summary.has_mcdc:
            mcdc_cell = (
                f"**{file_metric.mcdc_coverage_pct:.1f}%** ({file_metric.mcdc_cov}/{file_metric.mcdc_tot})"
                if file_metric.mcdc_tot > 0
                else "N/A"
            )
            dec_cell = (
                f"**{file_metric.decisions_ver} / {file_metric.decisions_tot}**"
                if file_metric.decisions_tot > 0
                else "N/A"
            )
            diag_cell = "<br>".join(file_metric.condition_diagnostics) if file_metric.condition_diagnostics else "None"
            lines.append(
                f"| {file_link} | **{file_metric.line_coverage_pct:.2f}%** ({f_cov_len}/{f_tot_len}) | {mcdc_cell} | {dec_cell} | {len(f_missed)} | {diag_cell} |"
            )
        else:
            line_spans = format_line_ranges(f_missed)
            lines.append(
                f"| {file_link} | **{file_metric.line_coverage_pct:.2f}%** | {f_cov_len} / {f_tot_len} | {len(f_missed)} | {line_spans} |"
            )

    # Summary Row
    if summary.has_mcdc:
        total_dec_cell = f"**{summary.fully_verified_decisions} / {summary.total_decisions_count}**"
        lines.append(
            f"| **Total (Patch)** | **{summary.line_coverage_pct:.2f}%** ({summary.total_covered_lines}/{summary.total_lines}) | **{summary.mcdc_coverage_pct:.1f}%** ({summary.total_mcdc_cov}/{summary.total_mcdc_tot}) | {total_dec_cell} | **{summary.total_missed_lines}** | - |"
        )
    else:
        lines.append(
            f"| **Total (Patch)** | **{summary.line_coverage_pct:.2f}%** | {summary.total_covered_lines} / {summary.total_lines} | **{summary.total_missed_lines}** | - |"
        )

    return "\n".join(lines)


def format_annotated_diff(
    summary: PatchCoverageSummary,
    diff_files: Dict[str, List[DiffHunk]],
) -> str:
    """Renders the collapsible source map diff with execution indicators."""
    lines: List[str] = [
        "<details>",
        "<summary><b>View Annotated Patch Diff (Source Map)</b></summary>\n",
    ]

    for fpath, file_metric in summary.files.items():
        hunks = diff_files.get(fpath, [])
        unverified_lines = file_metric.unverified_decision_lines

        lines.append(f"#### `{fpath}`")
        lines.append("```diff")
        for hunk in hunks:
            lines.append(hunk.header)
            for l_type, text, l_num in hunk.lines:
                if l_type == "+":
                    if l_num in file_metric.missed_lines:
                        lines.append(f"- {text}  // [MISSED]")
                    elif l_num in unverified_lines:
                        unverified_conds = ", ".join(unverified_lines[l_num])
                        lines.append(f"! {text}  // [PARTIAL MC/DC: {unverified_conds} unverified]")
                    elif l_num in file_metric.covered_lines:
                        lines.append(f"+ {text}")
                    else:
                        lines.append(f"  {text}")
                elif l_type == " ":
                    lines.append(f"  {text}")
        lines.append("```\n")

    lines.append("</details>")
    return "\n".join(lines)


def render_patch_report(
    diff_files: Dict[str, List[DiffHunk]],
    coverage_matrix: Dict[str, Dict[str, Any]],
    base_sha: Optional[str],
    head_sha: Optional[str],
    base_branch: Optional[str],
    head_branch: Optional[str],
    targets_str: Optional[str] = None,
    base_repo: str = DEFAULT_BASE_REPO,
    head_repo: str = DEFAULT_HEAD_REPO,
) -> None:
    """Composes and outputs the full Markdown report."""
    summary = calculate_patch_statistics(diff_files, coverage_matrix)

    if summary.has_mcdc:
        print("## LLVM-libc MC/DC Patch Coverage Report\n")
    else:
        print("## LLVM-libc Patch Coverage Report\n")

    if summary.total_lines == 0 or not summary.files:
        meta_str = format_metadata_section(
            base_sha, head_sha, base_branch, head_branch, targets_str, base_repo, head_repo
        )
        if meta_str:
            print(meta_str)
            print("\n---\n")
        print("> [!NOTE]")
        print("> ### Coverage Validated")
        print("> No `.cpp` source files in `libc/src/` were modified in this patch.")
        return

    # 1. Status Banner
    print(format_status_banner(summary))
    print("")

    # 2. Metadata Section
    meta_str = format_metadata_section(
        base_sha, head_sha, base_branch, head_branch, targets_str, base_repo, head_repo
    )
    if meta_str:
        print(meta_str)
        print("\n---\n")

    # 3. Breakdown Table
    print(format_breakdown_table(summary, head_repo, head_sha))
    print("")

    # 4. Source Map Diff
    print(format_annotated_diff(summary, diff_files))


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    """Parses command-line arguments and triggers report generation."""
    parser = argparse.ArgumentParser(description="LLVM-libc Diff Coverage Analyzer")
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
        default=DEFAULT_BASE_REPO,
        help=f"Base repository (default: {DEFAULT_BASE_REPO})",
    )
    parser.add_argument(
        "head_repo",
        nargs="?",
        default=DEFAULT_HEAD_REPO,
        help=f"Head repository (default: {DEFAULT_HEAD_REPO})",
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
