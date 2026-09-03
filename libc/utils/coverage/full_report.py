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
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def render_full_report(cov_data: dict) -> None:
    if "data" not in cov_data or not cov_data["data"]:
        print("## LLVM-libc Full Codebase Coverage Report\n")
        print("> [!WARNING]")
        print("> ### No Coverage Data Detected")
        print(
            "> The test execution completed but no coverage profiles were exported."
        )
        return

    subsystems: Dict[str, Dict[str, Any]] = {}

    total_lines_cov = 0
    total_lines_tot = 0
    total_func_cov = 0
    total_func_tot = 0
    total_mcdc_cov = 0
    total_mcdc_tot = 0
    total_decisions_count = 0
    total_decisions_full = 0

    for item in cov_data["data"][0].get("files", []):
        fpath = item.get("filename", "")
        if "src/" not in fpath or "/test/" in fpath or "/utils/" in fpath:
            continue

        idx = fpath.find("src/")
        if idx == -1:
            continue
        rel_path = fpath[idx:]

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

        mcdc_records = item.get("mcdc_records", [])
        file_decisions_tot = len(mcdc_records)
        file_decisions_full = 0
        for rec in mcdc_records:
            if len(rec) >= 10 and isinstance(rec[9], list):
                conds = rec[9]
                if all(conds):
                    file_decisions_full += 1

        if line_tot == 0:
            continue

        total_lines_cov += line_cov
        total_lines_tot += line_tot
        total_func_cov += func_cov
        total_func_tot += func_tot
        total_mcdc_cov += mcdc_cov
        total_mcdc_tot += mcdc_tot
        total_decisions_count += file_decisions_tot
        total_decisions_full += file_decisions_full

        parts = rel_path.split("/")
        subsystem = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]

        if subsystem not in subsystems:
            subsystems[subsystem] = {
                "lines_cov": 0,
                "lines_tot": 0,
                "func_cov": 0,
                "func_tot": 0,
                "mcdc_cov": 0,
                "mcdc_tot": 0,
                "decisions_tot": 0,
                "decisions_full": 0,
            }

        subsystems[subsystem]["lines_cov"] += line_cov
        subsystems[subsystem]["lines_tot"] += line_tot
        subsystems[subsystem]["func_cov"] += func_cov
        subsystems[subsystem]["func_tot"] += func_tot
        subsystems[subsystem]["mcdc_cov"] += mcdc_cov
        subsystems[subsystem]["mcdc_tot"] += mcdc_tot
        subsystems[subsystem]["decisions_tot"] += file_decisions_tot
        subsystems[subsystem]["decisions_full"] += file_decisions_full

    line_pct = (
        (total_lines_cov / total_lines_tot * 100) if total_lines_tot > 0 else 0
    )
    func_pct = (
        (total_func_cov / total_func_tot * 100) if total_func_tot > 0 else 0
    )
    has_mcdc = total_mcdc_tot > 0
    mcdc_pct = (
        (total_mcdc_cov / total_mcdc_tot * 100) if total_mcdc_tot > 0 else 0
    )
    decisions_pct = (
        (total_decisions_full / total_decisions_count * 100)
        if total_decisions_count > 0
        else 0
    )

    pages_url = os.environ.get("COVERAGE_DASHBOARD_URL")
    if not pages_url:
        repo = os.environ.get("GITHUB_REPOSITORY", "llvm/llvm-project")
        if "/" in repo:
            owner, repo_name = repo.split("/", 1)
            pages_url = f"https://{owner}.github.io/{repo_name}/"
        else:
            pages_url = f"https://{repo}.github.io/"

    if has_mcdc and not pages_url.endswith("/mcdc/"):
        mcdc_pages_url = pages_url.rstrip("/") + "/mcdc/"
    else:
        mcdc_pages_url = pages_url

    print("## LLVM-libc Full Codebase Coverage Report\n")

    print("> [!NOTE]")
    if has_mcdc:
        print(
            f"> ### Overall Codebase Coverage: **{line_pct:.2f}% Line** | **{mcdc_pct:.2f}% MC/DC**"
        )
        print(
            f"> Tested **{total_lines_cov:,} / {total_lines_tot:,}** executable lines and **{total_mcdc_cov:,} / {total_mcdc_tot:,}** boolean conditions across **{total_decisions_count:,}** decisions."
        )
        print("")
        print(
            f"- **Coverage Dashboard:** [{mcdc_pages_url}]({mcdc_pages_url})"
        )
    else:
        print(f"> ### Overall Codebase Coverage: **{line_pct:.2f}%**")
        print(
            f"> Tested **{total_lines_cov:,} / {total_lines_tot:,}** executable lines across all LLVM-libc subsystems."
        )
        print("")
        print(f"- **Coverage Dashboard:** [{pages_url}]({pages_url})")

    print("\n---\n")

    print("### Overall")
    print("| Metric | Covered | Total | Coverage % |")
    print("| :--- | :---: | :---: | :---: |")
    if has_mcdc:
        print(
            f"| **MC/DC Condition Independence** | {total_mcdc_cov:,} | {total_mcdc_tot:,} | **{mcdc_pct:.2f}%** |"
        )
        print(
            f"| **Fully Verified Decisions** | {total_decisions_full:,} | {total_decisions_count:,} | **{decisions_pct:.2f}%** |"
        )
    print(
        f"| **Executable Lines** | {total_lines_cov:,} | {total_lines_tot:,} | **{line_pct:.2f}%** |"
    )
    print(
        f"| **Functions** | {total_func_cov:,} | {total_func_tot:,} | **{func_pct:.2f}%** |"
    )
    print("")

    print("### Coverage Breakdown")
    if has_mcdc:
        print(
            "| Subsystem | MC/DC Conditions | Decisions (Verified / Total) | Line Coverage | Function Coverage | Executable Lines | Missed Lines |"
        )
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        sorted_subsystems = sorted(subsystems.keys())
    else:
        print(
            "| Subsystem | Line Coverage | Function Coverage | Executable Lines | Missed Lines |"
        )
        print("| :--- | :---: | :---: | :---: | :---: |")
        sorted_subsystems = sorted(subsystems.keys())

    for sub in sorted_subsystems:
        data = subsystems[sub]
        s_line_pct = (
            (data["lines_cov"] / data["lines_tot"] * 100)
            if data["lines_tot"] > 0
            else 0
        )
        s_func_pct = (
            (data["func_cov"] / data["func_tot"] * 100)
            if data["func_tot"] > 0
            else 0
        )
        missed_lines = data["lines_tot"] - data["lines_cov"]

        if has_mcdc:
            s_mc_pct = (
                (data["mcdc_cov"] / data["mcdc_tot"] * 100)
                if data["mcdc_tot"] > 0
                else 0
            )
            mc_cell = (
                f"**{s_mc_pct:.1f}%** ({data['mcdc_cov']}/{data['mcdc_tot']})"
                if data["mcdc_tot"] > 0
                else "N/A"
            )
            dec_cell = (
                f"{data['decisions_full']} / {data['decisions_tot']}"
                if data["decisions_tot"] > 0
                else "N/A"
            )
            print(
                f"| `libc/{sub}` | {mc_cell} | {dec_cell} | **{s_line_pct:.2f}%** | {s_func_pct:.2f}% | {data['lines_tot']:,} | {missed_lines:,} |"
            )
        else:
            print(
                f"| `libc/{sub}` | **{s_line_pct:.2f}%** | {s_func_pct:.2f}% | {data['lines_tot']:,} | {missed_lines:,} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLVM-libc Full Coverage Analyzer")
    parser.add_argument("json_file", help="Path to llvm-cov export JSON file")

    args, _ = parser.parse_known_args()

    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            cov_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Error: Failed to parse coverage JSON: {e}\n")
        sys.exit(1)
    render_full_report(cov_data)


if __name__ == "__main__":
    main()
