import subprocess
import sys
from pathlib import Path

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "inter-integration" if config.inter_test_is_integration else "inter"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".ll", ".mlir", ".py"]
if config.inter_test_is_integration:
    config.suffixes.append(".cl")
config.excludes = ["Inputs", "lit.cfg.py", "lit.site.cfg.py"]
if not config.inter_test_is_integration:
    config.excludes.append("Integration")

config.test_source_root = config.inter_test_source_root
config.test_exec_root = config.inter_test_exec_root

inter_pipelines = Path(config.inter_pipelines)
if inter_pipelines.exists():
    config.substitutions.append(("%inter_pipelines", str(inter_pipelines)))
else:
    lit_config.fatal(f"missing Inter pipeline library: {inter_pipelines}")
config.substitutions.append(("%python", f'"{sys.executable}"'))
config.substitutions.append(("%inter_obj_root", config.inter_obj_root))

llvm_config.with_system_environment(
    ["HOME", "INCLUDE", "LIB", "LD_LIBRARY_PATH", "TMP", "TEMP"]
)

tools = [
    ToolSubst("FileCheck", command=config.filecheck, unresolved="fatal"),
    ToolSubst("llvm-objcopy", command=config.llvm_objcopy, unresolved="fatal"),
    ToolSubst("llvm-readobj", command=config.llvm_readobj, unresolved="fatal"),
    ToolSubst("inter-opt", command=config.inter_opt, unresolved="fatal"),
    ToolSubst(
        "inter-alias-dump", command=config.inter_alias_dump, unresolved="fatal"
    ),
    ToolSubst(
        "inter-compile-api-test",
        command=config.inter_compile_api_test,
        unresolved="fatal",
    ),
    ToolSubst(
        "inter-timing-dump",
        command=config.inter_timing_dump,
        unresolved="fatal",
    ),
    ToolSubst(
        "inter-target-info",
        command=config.inter_target_info,
        unresolved="fatal",
    ),
    ToolSubst(
        "inter-translate", command=config.inter_translate, unresolved="fatal"
    ),
    ToolSubst(
        "inter-ged-dump", command=config.inter_ged_dump, unresolved="fatal"
    ),
]

if config.inter_test_is_integration:
    config.maxIndividualTestTime = 45
    lit_config.parallelism_groups["inter-xe2"] = 1
    config.parallelism_group = "inter-xe2"
    llvm_config.with_environment("INTER_DEVICE_NAME", config.inter_device_name)

    tools.extend(
        [
            ToolSubst(
                "inter-runner",
                command=config.inter_runner,
                unresolved="fatal",
            ),
            ToolSubst(
                "inter-matmul-runner",
                command=config.inter_matmul_runner,
                unresolved="fatal",
            ),
            ToolSubst("ocloc", command=config.inter_ocloc, unresolved="fatal"),
        ]
    )

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
