#!/usr/bin/env python3
#
# ===- Generate codebase coverage reports --------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

"""
Standalone file for generating whole-codebase statement, branch, and MC/DC
coverage reports.

This script parses full-codebase `llvm-cov export` JSON files, aggregates
metrics across all top-level LLVM-libc directories (e.g. `src/ctype`, `src/math`,
`src/string`), and outputs Markdown summary tables for CI step summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_REPOSITORY = "llvm/llvm-project"


@dataclass
class DirectoryCoverageMetrics:
    """Encapsulates coverage metrics and boolean decision counts for a directory."""

    name: str = ""
    lines_cov: int = 0
    lines_tot: int = 0
    func_cov: int = 0
    func_tot: int = 0
    mcdc_cov: int = 0
    mcdc_tot: int = 0
    decisions_tot: int = 0
    decisions_full: int = 0

    @property
    def line_pct(self) -> float:
        """Percentage of executed statements."""
        return (self.lines_cov / self.lines_tot * 100.0) if self.lines_tot > 0 else 0.0

    @property
    def func_pct(self) -> float:
        """Percentage of executed functions."""
        return (self.func_cov / self.func_tot * 100.0) if self.func_tot > 0 else 0.0

    @property
    def mcdc_pct(self) -> float:
        """Percentage of evaluated independent boolean conditions."""
        return (self.mcdc_cov / self.mcdc_tot * 100.0) if self.mcdc_tot > 0 else 0.0

    @property
    def decisions_pct(self) -> float:
        """Percentage of fully verified boolean decisions."""
        if self.decisions_tot == 0:
            return 0.0
        return self.decisions_full / self.decisions_tot * 100.0

    @property
    def missed_lines(self) -> int:
        """Count of unexecuted lines."""
        return max(0, self.lines_tot - self.lines_cov)


@dataclass
class FullCoverageSummary:
    """Encapsulates global and directory-level coverage statistics across LLVM-libc."""

    global_stats: DirectoryCoverageMetrics = field(
        default_factory=DirectoryCoverageMetrics
    )
    directories: Dict[str, DirectoryCoverageMetrics] = field(default_factory=dict)

    @property
    def has_mcdc(self) -> bool:
        """Returns True if any MC/DC condition data exists in the summary."""
        return self.global_stats.mcdc_tot > 0


def extract_full_coverage_statistics(
    cov_data: dict,
) -> Optional[FullCoverageSummary]:
    """Extracts global and per-directory metrics from llvm-cov export JSON data."""
    if "data" not in cov_data or not cov_data["data"]:
        return None

    global_metrics = DirectoryCoverageMetrics(name="global")
    directories: Dict[str, DirectoryCoverageMetrics] = {}

    for item in cov_data["data"][0].get("files", []):
        file_path = item.get("filename", "")
        if "src/" not in file_path or "/test/" in file_path or "/utils/" in file_path:
            continue

        idx = file_path.find("src/")
        rel_path = file_path[idx:]

        summary = item.get("summary", {})
        lines_summary = summary.get("lines", {})
        func_summary = summary.get("functions", {})
        mcdc_summary = summary.get("mcdc", {})

        line_tot = lines_summary.get("count", 0)
        line_cov = lines_summary.get("covered", 0)
        func_tot = func_summary.get("count", 0)
        func_cov = func_summary.get("covered", 0)
        mcdc_tot = mcdc_summary.get("count", 0)
        mcdc_cov = mcdc_summary.get("covered", 0)

        if line_tot == 0:
            continue

        mcdc_records = item.get("mcdc_records", [])
        valid_mcdc_records = [
            rec
            for rec in mcdc_records
            if len(rec) >= 10 and isinstance(rec[9], list) and len(rec[9]) > 0
        ]
        file_decisions_tot = len(valid_mcdc_records)
        file_decisions_full = sum(1 for rec in valid_mcdc_records if all(rec[9]))

        global_metrics.lines_cov += line_cov
        global_metrics.lines_tot += line_tot
        global_metrics.func_cov += func_cov
        global_metrics.func_tot += func_tot
        global_metrics.mcdc_cov += mcdc_cov
        global_metrics.mcdc_tot += mcdc_tot
        global_metrics.decisions_tot += file_decisions_tot
        global_metrics.decisions_full += file_decisions_full

        parts = rel_path.split("/")
        directory_name = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]

        if directory_name not in directories:
            directories[directory_name] = DirectoryCoverageMetrics(name=directory_name)

        dir_metrics = directories[directory_name]
        dir_metrics.lines_cov += line_cov
        dir_metrics.lines_tot += line_tot
        dir_metrics.func_cov += func_cov
        dir_metrics.func_tot += func_tot
        dir_metrics.mcdc_cov += mcdc_cov
        dir_metrics.mcdc_tot += mcdc_tot
        dir_metrics.decisions_tot += file_decisions_tot
        dir_metrics.decisions_full += file_decisions_full

    if global_metrics.lines_tot == 0:
        return None

    return FullCoverageSummary(
        global_stats=global_metrics,
        directories=directories,
    )


def format_overview_callout(summary: FullCoverageSummary) -> str:
    """Generates the executive summary banner."""
    g = summary.global_stats
    lines: List[str] = []

    if summary.has_mcdc:
        lines.append(
            f"### Overall Codebase Coverage: **{g.line_pct:.2f}% Line**"
            f" | **{g.mcdc_pct:.2f}% MC/DC**"
        )
        lines.append(
            f"Tested **{g.lines_cov:,} / {g.lines_tot:,}** executable lines "
            f"and **{g.mcdc_cov:,} / {g.mcdc_tot:,}** boolean conditions "
            f"across **{g.decisions_tot:,}** decisions."
        )
    else:
        lines.append(f"### Overall Codebase Coverage: **{g.line_pct:.2f}%**")
        lines.append(
            f"Tested **{g.lines_cov:,} / {g.lines_tot:,}** executable lines "
            "across all LLVM-libc directories."
        )

    lines.append("")
    lines.append(
        "- **HTML Coverage Report:** Available for download under the "
        "**Artifacts** section of this workflow run."
    )
    return "\n".join(lines)


def format_global_summary_table(summary: FullCoverageSummary) -> str:
    """Generates the top-level metric summary table."""
    g = summary.global_stats
    lines: List[str] = [
        "### Overall",
        "| Metric | Covered | Total | Coverage % |",
        "| :--- | :---: | :---: | :---: |",
    ]

    if summary.has_mcdc:
        lines.append(
            f"| **MC/DC Condition Independence** | {g.mcdc_cov:,} | "
            f"{g.mcdc_tot:,} | **{g.mcdc_pct:.2f}%** |"
        )
        lines.append(
            f"| **Fully Verified Decisions** | {g.decisions_full:,} | "
            f"{g.decisions_tot:,} | **{g.decisions_pct:.2f}%** |"
        )

    lines.append(
        f"| **Executable Lines** | {g.lines_cov:,} | {g.lines_tot:,} | "
        f"**{g.line_pct:.2f}%** |"
    )
    lines.append(
        f"| **Functions** | {g.func_cov:,} | {g.func_tot:,} | "
        f"**{g.func_pct:.2f}%** |"
    )
    return "\n".join(lines)


def format_directory_breakdown_table(summary: FullCoverageSummary) -> str:
    """Generates the directory breakdown table."""
    lines: List[str] = ["### Coverage Breakdown"]
    has_mcdc = summary.has_mcdc

    if has_mcdc:
        lines.append(
            "| Directory | MC/DC Conditions | Decisions (Verified / Total) | "
            "Line Coverage | Function Coverage | Executable Lines | Missed Lines |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    else:
        lines.append(
            "| Directory | Line Coverage | Function Coverage | "
            "Executable Lines | Missed Lines |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for dir_name in sorted(summary.directories.keys()):
        data = summary.directories[dir_name]
        if has_mcdc:
            mc_cell = (
                f"**{data.mcdc_pct:.1f}%** ({data.mcdc_cov}/{data.mcdc_tot})"
                if data.mcdc_tot > 0
                else "N/A"
            )
            dec_cell = (
                f"{data.decisions_full} / {data.decisions_tot}"
                if data.decisions_tot > 0
                else "N/A"
            )
            lines.append(
                f"| `libc/{dir_name}` | {mc_cell} | {dec_cell} | "
                f"**{data.line_pct:.2f}%** | {data.func_pct:.2f}% | "
                f"{data.lines_tot:,} | {data.missed_lines:,} |"
            )
        else:
            lines.append(
                f"| `libc/{dir_name}` | **{data.line_pct:.2f}%** | "
                f"{data.func_pct:.2f}% | {data.lines_tot:,} | "
                f"{data.missed_lines:,} |"
            )

    return "\n".join(lines)


def render_full_report(cov_data: dict) -> None:
    """Orchestrates extraction and renders the full Markdown report to stdout."""
    summary = extract_full_coverage_statistics(cov_data)

    print("## LLVM-libc Full Codebase Coverage Report\n")

    if not summary:
        print("### No Coverage Data Detected")
        print("The test execution completed but no coverage profiles were exported.")
        return

    # 1. Executive Callout Banner
    print(format_overview_callout(summary))
    print("\n---\n")

    # 2. Global Metric Summary Table
    print(format_global_summary_table(summary))
    print("")

    # 3. Directory Breakdown Table
    print(format_directory_breakdown_table(summary))


def main() -> None:
    """Parses command-line arguments and triggers report generation."""
    parser = argparse.ArgumentParser(description="LLVM-libc Codebase Coverage Analyzer")
    parser.add_argument("json_file", help="Path to llvm-cov export JSON file")
    parser.add_argument(
        "commit_sha",
        nargs="?",
        default="",
        help="Commit SHA under evaluation",
    )
    parser.add_argument(
        "branch_ref",
        nargs="?",
        default="",
        help="Branch reference under evaluation",
    )

    args, _ = parser.parse_known_args()

    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            cov_data = json.load(f)
    except Exception as err:
        sys.stderr.write(
            f"Error: Failed to parse coverage JSON from '{args.json_file}': {err}\n"
        )
        sys.exit(1)

    render_full_report(cov_data)


if __name__ == "__main__":
    main()
