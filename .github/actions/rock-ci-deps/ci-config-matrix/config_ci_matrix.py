#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build test CI matrix JSON with runtime runner selection."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _gha_output(name: str, value: str) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        raise RuntimeError("GITHUB_OUTPUT is not set")
    with open(out_path, "a", encoding="utf-8") as out:
        out.write(f"{name}={value}\n")


def main() -> None:
    workspace = Path.cwd()
    sys.path.insert(0, str(workspace / "build_tools" / "github_actions"))
    from amdgpu_family_matrix import select_build_runner

    variant = os.environ.get("BUILD_VARIANT", "release")
    prebuilt_stages = os.environ.get("PREBUILT_STAGES", "")
    baseline_run_id = os.environ.get("BASELINE_RUN_ID", "")

    linux_config = {
        "per_family_info": [
            {
                "amdgpu_family": "gfx94X-dcgpu",
                "amdgpu_targets": "gfx942",
                "test-runs-on": "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
                "sanity_check_only_for_family": False,
            },
            {
                "amdgpu_family": "gfx110X-all",
                "amdgpu_targets": "",
                "test-runs-on": "",
                "sanity_check_only_for_family": True,
                "bypass_tests_for_releases": True,
            },
            {
                "amdgpu_family": "gfx90a",
                "amdgpu_targets": "",
                "test-runs-on": "",
                "sanity_check_only_for_family": True,
                "bypass_tests_for_releases": True,
            },
            {
                "amdgpu_family": "gfx125X-dcgpu",
                "amdgpu_targets": "gfx1250",
                "test-runs-on": "",
                "sanity_check_only_for_family": True,
                "bypass_tests_for_releases": True,
            },
        ],
        "dist_amdgpu_families": "gfx94X-dcgpu;gfx110X-all;gfx90a;gfx125X-dcgpu",
        "artifact_group": "multi-arch-release",
        "build_variant_label": "release",
        "build_variant_suffix": "",
        "build_variant_cmake_preset": "",
        "build_python_packages": True,
        "build_pytorch": True,
        "build_rocm": True,
        "build_runs_on": "aws-linux-scale-rocm-large",
        "prebuilt_stages": prebuilt_stages,
        "baseline_run_id": baseline_run_id,
    }

    windows_config = {
        "per_family_info": [
            {
                "amdgpu_family": "gfx1151",
                "amdgpu_targets": "",
                "test-runs-on": "",
                "benchmark-runs-on": "",
                "sanity_check_only_for_family": False,
            },
        ],
        "dist_amdgpu_families": "gfx1151",
        "artifact_group": "multi-arch-release",
        "build_variant_label": "release",
        "build_variant_suffix": "",
        "build_variant_cmake_preset": "windows-release",
        "build_python_packages": True,
        "build_pytorch": True,
        "build_rocm": True,
        "build_runs_on": select_build_runner("windows", variant),
        "prebuilt_stages": prebuilt_stages,
        "baseline_run_id": baseline_run_id,
    }

    _gha_output("linux_build_config", json.dumps(linux_config))
    _gha_output("windows_build_config", json.dumps(windows_config))

    print("Linux build_runs_on:", linux_config["build_runs_on"])
    print("Windows build_runs_on:", windows_config["build_runs_on"])


if __name__ == "__main__":
    main()
