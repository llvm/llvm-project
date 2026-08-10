import os
import subprocess
import sys
from pathlib import Path

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "inter-integration" if config.inter_test_is_integration else "inter"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".ll"] if config.inter_test_is_integration else [".mlir"]
config.excludes = ["Inputs", "lit.cfg.py", "lit.site.cfg.py"]
if not config.inter_test_is_integration:
    config.excludes.append("Integration")

config.test_source_root = config.inter_test_source_root
config.test_exec_root = config.inter_test_exec_root

llvm_config.with_system_environment(
    ["HOME", "INCLUDE", "LIB", "LD_LIBRARY_PATH", "TMP", "TEMP"]
)

tools = [
    ToolSubst("FileCheck", command=config.filecheck, unresolved="fatal"),
    ToolSubst("inter-opt", command=config.inter_opt, unresolved="fatal"),
    ToolSubst(
        "inter-translate", command=config.inter_translate, unresolved="fatal"
    ),
    ToolSubst(
        "inter-ged-dump", command=config.inter_ged_dump, unresolved="fatal"
    ),
]

config.substitutions.append(("%python", f'"{sys.executable}"'))
config.substitutions.append(
    ("%make-zebin", f'"{Path(config.inter_src_root) / "make_zebin.py"}"')
)
if config.inter_test_is_integration:
    config.maxIndividualTestTime = 45
    lit_config.parallelism_groups["inter-xe2"] = 1
    config.parallelism_group = "inter-xe2"
    llvm_config.with_environment("INTER_DEVICE_NAME", config.inter_device_name)

    tools.extend(
        [
            ToolSubst("%ocloc", command=config.inter_ocloc, unresolved="fatal"),
            ToolSubst(
                "inter-runner",
                command=config.inter_runner,
                unresolved="fatal",
            ),
        ]
    )
    config.substitutions.append(("%inter-device", config.inter_ocloc_device))

    try:
        probe = subprocess.run(
            [config.inter_runner, "--probe", config.inter_device_name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=config.environment,
        )
    except (OSError, subprocess.SubprocessError):
        probe = None
    if probe and probe.returncode == 0:
        config.available_features.add("host-supports-inter-bmg")

llvm_config.add_tool_substitutions(tools, [])
