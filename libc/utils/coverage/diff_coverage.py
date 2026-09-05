#!/usr/bin/env python3
#
# ===- Generate diff coverage reports ------------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

"""
Standalone analyzer for evaluating diff-level statement, branch, and MC/DC
coverage.

This script parses unified git diffs alongside `llvm-cov export` JSON summaries,
correlates added/modified lines with execution counts and boolean decision
records, and outputs formatted Markdown reports for CI job summaries and PR
comments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_BASE_REPOSITORY = "llvm/llvm-project"
DEFAULT_HEAD_REPOSITORY = "llvm/llvm-project"

COMMENT_PREFIXES = ("//", "/*", "*", "*/")
STRUCTURAL_TOKENS = ("{", "}", "};", "{};")
DECLARATION_PREFIXES = (
    "namespace ",
    "extern ",
    "using ",
    "__attribute__",
    "template",
    "typedef ",
    "friend ",
    "static_assert",
)
ACCESS_SPECIFIERS = ("public:", "private:", "protected:")


@dataclass
class DiffHunk:
    """Represents a unified diff hunk with its header and line tokens."""

    header: str
    lines: List[Tuple[str, str, int]] = field(
        default_factory=list
    )  # (prefix, text, line_number)


@dataclass
class FilePatchMetrics:
    """Encapsulates coverage metrics and decisions for a single modified file."""

    file_path: str
    covered_lines: Set[int] = field(default_factory=set)
    missed_lines: Set[int] = field(default_factory=set)
    added_lines: Set[int] = field(default_factory=set)
    mcdc_covered_conditions: int = 0
    mcdc_total_conditions: int = 0
    decisions_verified: int = 0
    decisions_total: int = 0
    condition_diagnostics: List[str] = field(default_factory=list)
    unverified_decision_lines: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def total_lines(self) -> int:
        """Total executable lines evaluated in this file."""
        return len(self.covered_lines) + len(self.missed_lines)

    @property
    def line_coverage_percentage(self) -> float:
        """Percentage of executed patch lines."""
        return (
            (len(self.covered_lines) / self.total_lines * 100.0)
            if self.total_lines > 0
            else 0.0
        )

    @property
    def mcdc_coverage_percentage(self) -> float:
        """Percentage of independent boolean conditions evaluated."""
        return (
            (self.mcdc_covered_conditions / self.mcdc_total_conditions * 100.0)
            if self.mcdc_total_conditions > 0
            else 0.0
        )


@dataclass
class PatchCoverageSummary:
    """Aggregated coverage statistics across all modified files in the patch."""

    files: Dict[str, FilePatchMetrics] = field(default_factory=dict)
    total_covered_lines: int = 0
    total_missed_lines: int = 0
    total_mcdc_covered_conditions: int = 0
    total_mcdc_total_conditions: int = 0
    total_decisions_count: int = 0
    fully_verified_decisions: int = 0

    @property
    def total_lines(self) -> int:
        """Total executable lines across all modified files in the patch."""
        return self.total_covered_lines + self.total_missed_lines

    @property
    def line_coverage_percentage(self) -> float:
        """Aggregated patch line coverage percentage."""
        return (
            (self.total_covered_lines / self.total_lines * 100.0)
            if self.total_lines > 0
            else 0.0
        )

    @property
    def mcdc_coverage_percentage(self) -> float:
        """Aggregated patch MC/DC condition coverage percentage."""
        if self.total_mcdc_total_conditions == 0:
            return 0.0
        return (
            self.total_mcdc_covered_conditions
            / self.total_mcdc_total_conditions
            * 100.0
        )

    @property
    def has_mcdc(self) -> bool:
        """Returns True if any MC/DC decision records intersect the patch."""
        return self.total_mcdc_total_conditions > 0


class DiffParser:
    """Parses unified diff outputs into structured file hunks with line numbers."""

    @staticmethod
    def parse(diff_source: str) -> Dict[str, List[DiffHunk]]:
        """Parses a diff file into a mapping of file path to hunks."""
        files: Dict[str, List[DiffHunk]] = {}
        current_file: Optional[str] = None
        current_hunk: Optional[DiffHunk] = None
        current_line_number: int = 0

        if os.path.isfile(diff_source):
            with open(
                diff_source, "r", encoding="utf-8", errors="replace"
            ) as file_handle:
                lines = file_handle.readlines()
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
                hunk_match = re.search(r"\+([0-9]+)", line)
                if hunk_match:
                    current_line_number = int(hunk_match.group(1))
                    current_hunk = DiffHunk(header=line)
                    files[current_file].append(current_hunk)
                continue

            if current_hunk is None:
                continue

            if line.startswith("-"):
                continue
            elif line.startswith("+"):
                current_hunk.lines.append(("+", line[1:], current_line_number))
                current_line_number += 1
            elif line.startswith(" "):
                current_hunk.lines.append((" ", line[1:], current_line_number))
                current_line_number += 1

        return files


class CoverageJSONParser:
    """Parses statement segments and MC/DC records from llvm-cov JSON export."""

    @staticmethod
    def load(json_path: str) -> dict:
        """Loads JSON file from disk with error reporting."""
        try:
            with open(json_path, "r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except Exception as error:
            sys.stderr.write(
                f"Error: Failed to parse coverage JSON from '{json_path}': {error}\n"
            )
            sys.exit(1)

    @staticmethod
    def extract_patch_matrix(
        coverage_data: dict, diff_files: Dict[str, List[DiffHunk]]
    ) -> Dict[str, Dict[str, Any]]:
        """Maps coverage segments and MC/DC decision records to modified files."""
        coverage_matrix: Dict[str, Dict[str, Any]] = {
            file_path: {
                "covered": set(),
                "missed": set(),
                "mcdc_decisions": [],
            }
            for file_path in diff_files.keys()
        }

        if "data" not in coverage_data or not coverage_data["data"]:
            return coverage_matrix

        for item in coverage_data["data"][0].get("files", []):
            file_name = item.get("filename", "")
            relative_file_path = next(
                (
                    target_path
                    for target_path in diff_files.keys()
                    if file_name == target_path
                    or file_name.endswith("/" + target_path)
                    or target_path.endswith("/" + file_name)
                ),
                None,
            )
            if not relative_file_path:
                continue

            # 1. Process statement coverage segments
            segments = item.get("segments", [])
            for index, current_segment in enumerate(segments):
                line_start = current_segment[0]
                execution_count = current_segment[2]
                has_execution_count = current_segment[3]

                if not has_execution_count:
                    continue

                if index < len(segments) - 1:
                    next_segment = segments[index + 1]
                    next_line = next_segment[0]
                    end_range = next_line if next_line > line_start else line_start + 1
                else:
                    end_range = line_start + 1

                for line_number in range(line_start, end_range):
                    if execution_count > 0:
                        coverage_matrix[relative_file_path]["covered"].add(line_number)
                    else:
                        coverage_matrix[relative_file_path]["missed"].add(line_number)

            # 2. Process MC/DC decision records
            mcdc_records = item.get("mcdc_records", [])
            for record in mcdc_records:
                if (
                    len(record) >= 10
                    and isinstance(record[9], list)
                    and len(record[9]) > 0
                ):
                    decision_start_line = record[0]
                    decision_end_line = record[2]
                    boolean_conditions = record[9]
                    covered_conditions_count = sum(
                        1 for condition in boolean_conditions if condition
                    )
                    coverage_matrix[relative_file_path]["mcdc_decisions"].append(
                        {
                            "line_start": decision_start_line,
                            "line_end": decision_end_line,
                            "conditions": boolean_conditions,
                            "covered": covered_conditions_count,
                            "total": len(boolean_conditions),
                        }
                    )

        return coverage_matrix


def is_executable_line(line_text: str) -> bool:
    """Filters out non-executable code lines (comments, braces, pure declarations)."""
    stripped_line = line_text.strip()
    if not stripped_line:
        return False
    # Strip block comments on the same line and trailing line comments
    clean_line = re.sub(r"/\*.*?\*/", "", stripped_line)
    clean_line = re.sub(r"//.*$", "", clean_line).strip()
    if not clean_line:
        return False
    if (
        clean_line.startswith("*")
        or clean_line.startswith("/*")
        or clean_line.startswith("*/")
    ):
        return False
    if clean_line in STRUCTURAL_TOKENS or clean_line.startswith(":"):
        return False
    if clean_line in ACCESS_SPECIFIERS:
        return False
    if clean_line.startswith("#"):
        return False
    if any(clean_line.startswith(prefix) for prefix in DECLARATION_PREFIXES):
        return False
    if (
        clean_line.startswith("struct ")
        or clean_line.startswith("class ")
        or clean_line.startswith("enum ")
    ):
        if ("{" in clean_line and "=" not in clean_line) or (
            clean_line.endswith(";") and "=" not in clean_line and "(" not in clean_line
        ):
            return False
    return True


def format_line_ranges(line_numbers: Set[int]) -> str:
    """Formats an integer set of line numbers into concise span representations."""
    if not line_numbers:
        return "None"
    sorted_line_numbers = sorted(line_numbers)
    formatted_ranges: List[str] = []
    start_line = sorted_line_numbers[0]
    end_line = sorted_line_numbers[0]
    for current_number in sorted_line_numbers[1:]:
        if current_number == end_line + 1:
            end_line = current_number
        else:
            range_label = (
                f"`L{start_line}-L{end_line}`"
                if start_line != end_line
                else f"`L{start_line}`"
            )
            formatted_ranges.append(range_label)
            start_line = end_line = current_number
    range_label = (
        f"`L{start_line}-L{end_line}`" if start_line != end_line else f"`L{start_line}`"
    )
    formatted_ranges.append(range_label)
    return ", ".join(formatted_ranges)


def calculate_patch_statistics(
    diff_files: Dict[str, List[DiffHunk]],
    coverage_matrix: Dict[str, Dict[str, Any]],
) -> PatchCoverageSummary:
    """Calculates line, branch, and MC/DC statistics for modified patch files."""
    summary = PatchCoverageSummary()

    for file_path, file_data in coverage_matrix.items():
        if (
            not any(file_path.endswith(ext) for ext in (".cpp", ".c", ".h", ".inc"))
            or "/test/" in file_path
            or file_path.startswith("test/")
            or "/utils/" in file_path
            or file_path.startswith("utils/")
        ):
            continue

        added_lines: Set[int] = set()
        for hunk in diff_files.get(file_path, []):
            for line_type, text, line_number in hunk.lines:
                if line_type == "+" and is_executable_line(text):
                    added_lines.add(line_number)

        if not added_lines:
            continue

        file_covered_lines = added_lines.intersection(file_data["covered"])
        file_missed_lines = (
            added_lines.intersection(file_data["missed"])
        ) - file_covered_lines

        file_metrics = FilePatchMetrics(
            file_path=file_path,
            added_lines=added_lines,
        )

        if len(file_data["covered"]) > 0 or len(file_data["missed"]) > 0:
            file_metrics.covered_lines = file_covered_lines
            file_metrics.missed_lines = file_missed_lines
            summary.total_covered_lines += len(file_covered_lines)
            summary.total_missed_lines += len(file_missed_lines)
        else:
            file_metrics.missed_lines = added_lines
            summary.total_missed_lines += len(added_lines)

        # Evaluate MC/DC decision records intersecting modified lines
        for decision in file_data.get("mcdc_decisions", []):
            decision_start_line = decision["line_start"]
            decision_end_line = decision["line_end"]
            if any(
                decision_start_line <= line_number <= decision_end_line
                for line_number in added_lines
            ):
                summary.total_decisions_count += 1
                file_metrics.decisions_total += 1
                file_metrics.mcdc_covered_conditions += decision["covered"]
                file_metrics.mcdc_total_conditions += decision["total"]
                summary.total_mcdc_covered_conditions += decision["covered"]
                summary.total_mcdc_total_conditions += decision["total"]

                if decision["covered"] == decision["total"]:
                    summary.fully_verified_decisions += 1
                    file_metrics.decisions_verified += 1
                    file_metrics.condition_diagnostics.append(
                        f"`L{decision_start_line}`: "
                        f"{decision['covered']}/{decision['total']} verified"
                    )
                else:
                    uncovered_indices = [
                        f"C{condition_index + 1}"
                        for condition_index, is_covered in enumerate(
                            decision["conditions"]
                        )
                        if not is_covered
                    ]
                    unverified_conditions_string = ", ".join(uncovered_indices)
                    file_metrics.condition_diagnostics.append(
                        f"`L{decision_start_line}`: "
                        f"{decision['covered']}/{decision['total']} verified "
                        f"({unverified_conditions_string} unverified)"
                    )
                    for decision_line in range(
                        decision_start_line, decision_end_line + 1
                    ):
                        if decision_line in added_lines:
                            file_metrics.unverified_decision_lines[
                                decision_line
                            ] = uncovered_indices

        summary.files[file_path] = file_metrics

    return summary


def format_status_banner(summary: PatchCoverageSummary) -> str:
    """Generates the executive summary block."""
    lines: List[str] = []
    if summary.total_missed_lines == 0:
        if not summary.has_mcdc:
            lines.append(
                f"### Patch Coverage: **{summary.line_coverage_percentage:.2f}%**"
            )
            lines.append(
                f"All **{summary.total_lines}** newly added or modified "
                "executable lines are covered."
            )
        elif (
            summary.total_mcdc_covered_conditions == summary.total_mcdc_total_conditions
        ):
            lines.append(
                f"### Patch Coverage: "
                f"**{summary.line_coverage_percentage:.2f}% Line** | "
                "**100.00% MC/DC**"
            )
            lines.append(
                f"All **{summary.total_lines}** executable lines and "
                f"**{summary.total_mcdc_total_conditions}** boolean conditions "
                f"across **{summary.total_decisions_count}** decisions are covered."
            )
        else:
            lines.append(
                f"### Patch Coverage: "
                f"**{summary.line_coverage_percentage:.2f}% Line** | "
                f"**{summary.mcdc_coverage_percentage:.1f}% MC/DC**"
            )
            lines.append(
                f"Executed **{summary.total_covered_lines} / {summary.total_lines}** "
                f"lines. **{summary.total_mcdc_covered_conditions} / "
                f"{summary.total_mcdc_total_conditions}** boolean conditions "
                f"achieved independence across **{summary.fully_verified_decisions} / "
                f"{summary.total_decisions_count}** decisions."
            )
    else:
        if not summary.has_mcdc:
            lines.append(
                f"### Patch Coverage: "
                f"**{summary.line_coverage_percentage:.2f}%** "
                f"({summary.total_missed_lines} Missed Lines)"
            )
            lines.append(
                f"Executed **{summary.total_covered_lines} / {summary.total_lines}** "
                f"lines (**{summary.total_missed_lines}** unexecuted lines "
                "detected in patch)."
            )
        else:
            lines.append(
                f"### Patch Coverage: "
                f"**{summary.line_coverage_percentage:.2f}% Line** | "
                f"**{summary.mcdc_coverage_percentage:.1f}% MC/DC** "
                f"({summary.total_missed_lines} Missed Lines)"
            )
            lines.append(
                f"Executed **{summary.total_covered_lines} / {summary.total_lines}** "
                f"lines. **{summary.total_mcdc_covered_conditions} / "
                f"{summary.total_mcdc_total_conditions}** boolean conditions "
                f"achieved independence across **{summary.fully_verified_decisions} / "
                f"{summary.total_decisions_count}** decisions "
                f"(**{summary.total_missed_lines}** unexecuted lines "
                "detected in patch)."
            )
    return "\n".join(lines)


def format_metadata_section(
    base_commit_sha: Optional[str],
    head_commit_sha: Optional[str],
    base_branch_name: Optional[str],
    head_branch_name: Optional[str],
    targeted_tests_string: Optional[str] = None,
    base_repository: str = DEFAULT_BASE_REPOSITORY,
    head_repository: str = DEFAULT_HEAD_REPOSITORY,
) -> str:
    """Formats Git commit and target test metadata."""
    lines: List[str] = []
    if base_commit_sha and head_commit_sha and base_branch_name and head_branch_name:
        lines.append(
            f"- **Base Branch:** [`{base_branch_name}` ({base_commit_sha[:7]})]"
            f"(https://github.com/{base_repository}/commit/{base_commit_sha})"
        )
        lines.append(
            f"- **Head Commit:** [`{head_branch_name}` ({head_commit_sha[:7]})]"
            f"(https://github.com/{head_repository}/commit/{head_commit_sha})"
        )
    if targeted_tests_string:
        formatted_targets = ", ".join(
            f"`{target.strip()}`"
            for target in targeted_tests_string.split()
            if target.strip()
        )
        lines.append(f"- **Targeted Tests Executed:** {formatted_targets}")
    return "\n".join(lines)


def format_breakdown_table(
    summary: PatchCoverageSummary,
    head_repository: str = DEFAULT_HEAD_REPOSITORY,
    head_commit_sha: Optional[str] = None,
) -> str:
    """Generates the Markdown table breaking down coverage per source file."""
    lines: List[str] = ["### Coverage Breakdown"]
    if summary.has_mcdc:
        lines.append(
            "| Modified Source File | Line Coverage | MC/DC Conditions | "
            "Decisions (Verified / Total) | Missed Lines | Unverified Conditions |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    else:
        lines.append(
            "| Modified Source File | Patch Coverage | Covered / Total | "
            "Missed Lines | Unexecuted Line Spans |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for file_path, file_metric in summary.files.items():
        commit_ref = head_commit_sha or "main"
        repo_file_path = (
            file_path if file_path.startswith("libc/") else f"libc/{file_path}"
        )
        file_link = (
            f"[`{file_path}`]"
            f"(https://github.com/{head_repository}/blob/{commit_ref}/{repo_file_path})"
        )
        missed_lines = file_metric.missed_lines
        covered_count = len(file_metric.covered_lines)
        total_file_lines = file_metric.total_lines

        if summary.has_mcdc:
            mc_pct = file_metric.mcdc_coverage_percentage
            mc_cov = file_metric.mcdc_covered_conditions
            mc_tot = file_metric.mcdc_total_conditions
            mcdc_cell = (
                f"**{mc_pct:.1f}%** ({mc_cov}/{mc_tot})" if mc_tot > 0 else "N/A"
            )
            dec_cell = (
                f"**{file_metric.decisions_verified} / "
                f"{file_metric.decisions_total}**"
                if file_metric.decisions_total > 0
                else "N/A"
            )
            diagnostic_cell = (
                "<br>".join(file_metric.condition_diagnostics)
                if file_metric.condition_diagnostics
                else "None"
            )
            lines.append(
                f"| {file_link} | "
                f"**{file_metric.line_coverage_percentage:.2f}%** "
                f"({covered_count}/{total_file_lines}) | "
                f"{mcdc_cell} | {dec_cell} | "
                f"{len(missed_lines)} | {diagnostic_cell} |"
            )
        else:
            line_spans = format_line_ranges(missed_lines)
            lines.append(
                f"| {file_link} | "
                f"**{file_metric.line_coverage_percentage:.2f}%** | "
                f"{covered_count} / {total_file_lines} | "
                f"{len(missed_lines)} | {line_spans} |"
            )

    # Summary Row
    if summary.has_mcdc:
        total_decision_cell = (
            f"**{summary.fully_verified_decisions} / "
            f"{summary.total_decisions_count}**"
        )
        lines.append(
            f"| **Total (Patch)** | "
            f"**{summary.line_coverage_percentage:.2f}%** "
            f"({summary.total_covered_lines}/{summary.total_lines}) | "
            f"**{summary.mcdc_coverage_percentage:.1f}%** "
            f"({summary.total_mcdc_covered_conditions}/"
            f"{summary.total_mcdc_total_conditions}) | "
            f"{total_decision_cell} | **{summary.total_missed_lines}** | - |"
        )
    else:
        lines.append(
            f"| **Total (Patch)** | "
            f"**{summary.line_coverage_percentage:.2f}%** | "
            f"{summary.total_covered_lines} / {summary.total_lines} | "
            f"**{summary.total_missed_lines}** | - |"
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

    for file_path, file_metric in summary.files.items():
        hunks = diff_files.get(file_path, [])
        unverified_decision_lines = file_metric.unverified_decision_lines

        lines.append(f"#### `{file_path}`")
        lines.append("```diff")
        for hunk in hunks:
            lines.append(hunk.header)
            for line_type, line_text, line_number in hunk.lines:
                if line_type == "+":
                    if line_number in file_metric.missed_lines:
                        lines.append(f"- {line_text}  // [MISSED]")
                    elif line_number in unverified_decision_lines:
                        unverified_conditions = ", ".join(
                            unverified_decision_lines[line_number]
                        )
                        lines.append(
                            f"! {line_text}  // [PARTIAL MC/DC: "
                            f"{unverified_conditions} unverified]"
                        )
                    elif line_number in file_metric.covered_lines:
                        lines.append(f"+ {line_text}")
                    else:
                        lines.append(f"  {line_text}")
                elif line_type == " ":
                    lines.append(f"  {line_text}")
        lines.append("```\n")

    lines.append("</details>")
    return "\n".join(lines)


def render_patch_report(
    diff_files: Dict[str, List[DiffHunk]],
    coverage_matrix: Dict[str, Dict[str, Any]],
    base_commit_sha: Optional[str],
    head_commit_sha: Optional[str],
    base_branch_name: Optional[str],
    head_branch_name: Optional[str],
    targeted_tests_string: Optional[str] = None,
    base_repository: str = DEFAULT_BASE_REPOSITORY,
    head_repository: str = DEFAULT_HEAD_REPOSITORY,
) -> None:
    """Composes and outputs the full Markdown report."""
    summary = calculate_patch_statistics(diff_files, coverage_matrix)

    if summary.has_mcdc:
        print("## LLVM-libc MC/DC Patch Coverage Report\n")
    else:
        print("## LLVM-libc Patch Coverage Report\n")

    if summary.total_lines == 0 or not summary.files:
        metadata_section_string = format_metadata_section(
            base_commit_sha,
            head_commit_sha,
            base_branch_name,
            head_branch_name,
            targeted_tests_string,
            base_repository,
            head_repository,
        )
        if metadata_section_string:
            print(metadata_section_string)
            print("\n---\n")
        print("### Coverage Summary")
        print("No executable lines were added or modified in this patch.")
        return

    # 1. Status Banner
    print(format_status_banner(summary))
    print("")

    # 2. Metadata Section
    metadata_section_string = format_metadata_section(
        base_commit_sha,
        head_commit_sha,
        base_branch_name,
        head_branch_name,
        targeted_tests_string,
        base_repository,
        head_repository,
    )
    if metadata_section_string:
        print(metadata_section_string)
        print("\n---\n")

    # 3. Breakdown Table
    print(format_breakdown_table(summary, head_repository, head_commit_sha))
    print("")

    # 4. Source Map Diff
    print(format_annotated_diff(summary, diff_files))


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
        "targets",
        nargs="?",
        help="Space-separated list of executed test targets",
    )
    parser.add_argument(
        "base_repo",
        nargs="?",
        default=DEFAULT_BASE_REPOSITORY,
        help=f"Base repository (default: {DEFAULT_BASE_REPOSITORY})",
    )
    parser.add_argument(
        "head_repo",
        nargs="?",
        default=DEFAULT_HEAD_REPOSITORY,
        help=f"Head repository (default: {DEFAULT_HEAD_REPOSITORY})",
    )

    arguments = parser.parse_args()

    if not os.path.isfile(arguments.diff_file):
        sys.stderr.write(f"Error: Diff file not found: '{arguments.diff_file}'\n")
        sys.exit(1)

    diff_files = DiffParser.parse(arguments.diff_file)
    coverage_data = CoverageJSONParser.load(arguments.json_file)
    coverage_matrix = CoverageJSONParser.extract_patch_matrix(coverage_data, diff_files)

    render_patch_report(
        diff_files,
        coverage_matrix,
        arguments.base_sha,
        arguments.head_sha,
        arguments.base_branch,
        arguments.head_branch,
        arguments.targets,
        arguments.base_repo,
        arguments.head_repo,
    )


if __name__ == "__main__":
    main()
