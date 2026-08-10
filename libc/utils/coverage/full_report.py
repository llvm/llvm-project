#!/usr/bin/env python3
#
# ====- Generate full codebase coverage reports ----------------*- python -*--==#
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
from typing import Dict, List, Tuple

# Ensure local module import works regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from parser import CoverageJSONParser


def render_full_report(cov_data: dict) -> None:
    if "data" not in cov_data or not cov_data["data"]:
        print("## LLVM-libc Full Codebase Coverage Report\n")
        print("> [!WARNING]")
        print("> ### No Coverage Data Detected")
        print("> The test execution completed but no coverage profiles were exported.")
        return

    subsystems: Dict[str, Dict[str, int]] = {}
    file_stats: List[Tuple[str, int, int, int, int]] = []

    total_lines_cov = 0
    total_lines_tot = 0
    total_func_cov = 0
    total_func_tot = 0

    for item in cov_data["data"][0].get("files", []):
        fpath = item["filename"]
        if "src/" not in fpath or "/test/" in fpath or "/utils/" in fpath:
            continue

        idx = fpath.find("src/")
        if idx == -1:
            continue
        rel_path = fpath[idx:]

        summary = item.get("summary", {})
        lines_summary = summary.get("lines", {})
        func_summary = summary.get("functions", {})

        line_tot = lines_summary.get("count", 0)
        line_cov = lines_summary.get("covered", 0)
        func_tot = func_summary.get("count", 0)
        func_cov = func_summary.get("covered", 0)

        if line_tot == 0:
            continue

        total_lines_cov += line_cov
        total_lines_tot += line_tot
        total_func_cov += func_cov
        total_func_tot += func_tot

        parts = rel_path.split("/")
        subsystem = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]

        if subsystem not in subsystems:
            subsystems[subsystem] = {
                "lines_cov": 0,
                "lines_tot": 0,
                "func_cov": 0,
                "func_tot": 0,
            }

        subsystems[subsystem]["lines_cov"] += line_cov
        subsystems[subsystem]["lines_tot"] += line_tot
        subsystems[subsystem]["func_cov"] += func_cov
        subsystems[subsystem]["func_tot"] += func_tot

        file_stats.append((rel_path, line_cov, line_tot, func_cov, func_tot))

    line_pct = (total_lines_cov / total_lines_tot * 100) if total_lines_tot > 0 else 0
    func_pct = (total_func_cov / total_func_tot * 100) if total_func_tot > 0 else 0

    repo = os.environ.get("GITHUB_REPOSITORY", "tapiwagonga/llvm-project")
    if "/" in repo:
        owner, repo_name = repo.split("/", 1)
        pages_url = f"https://{owner}.github.io/{repo_name}/"
    else:
        pages_url = f"https://{repo}.github.io/"

    print("## LLVM-libc Full Codebase Coverage Report\n")

    print("> [!NOTE]")
    print(f"> ### Overall Codebase Coverage: **{line_pct:.2f}%**")
    print(
        f"> Successfully tested **{total_lines_cov:,} / {total_lines_tot:,}** executable lines across all LLVM-libc subsystems."
    )
    print("")

    print(f"- **Coverage Dashboard:** [{pages_url}]({pages_url})")
    print("\n---\n")

    print("### Codebase Health Metrics")
    print("| Metric | Covered | Total | Coverage % |")
    print("| :--- | :---: | :---: | :---: |")
    print(
        f"| **Executable Line Coverage** | {total_lines_cov:,} | {total_lines_tot:,} | **{line_pct:.2f}%** |"
    )
    print(
        f"| **Function Coverage** | {total_func_cov:,} | {total_func_tot:,} | **{func_pct:.2f}%** |\n"
    )

    print("### Subsystem Coverage Breakdown")
    print("| Subsystem | Line Coverage | Function Coverage | Executable Lines | Missed Lines |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for sub, data in sorted(subsystems.items()):
        s_line_pct = (data["lines_cov"] / data["lines_tot"] * 100) if data["lines_tot"] > 0 else 0
        s_func_pct = (data["func_cov"] / data["func_tot"] * 100) if data["func_tot"] > 0 else 0
        missed = data["lines_tot"] - data["lines_cov"]
        print(
            f"| `libc/{sub}` | **{s_line_pct:.2f}%** | {s_func_pct:.2f}% | {data['lines_tot']:,} | {missed:,} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLVM-libc Full Coverage Analyzer")
    parser.add_argument("json_file", help="Path to llvm-cov export JSON file")

    args, _ = parser.parse_known_args()

    cov_data = CoverageJSONParser.load(args.json_file)
    render_full_report(cov_data)


if __name__ == "__main__":
    main()
