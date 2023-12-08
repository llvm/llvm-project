# -*- Python -*-

import os

# Setup config name.
config.name = "ORC" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Determine whether the test target is compatible with execution on the host.
host_arch_compatible = config.target_arch == config.host_arch

if config.host_arch == "x86_64h" and config.target_arch == "x86_64":
    host_arch_compatible = True
config.test_target_is_host_executable = (
    config.target_os == config.host_os and host_arch_compatible
)

# Assume that llvm-jitlink is in the config.llvm_tools_dir.
llvm_jitlink = os.path.join(config.llvm_tools_dir, "llvm-jitlink")
orc_rt_executor_stem = os.path.join(
    config.compiler_rt_obj_root, "lib/orc/tests/tools/orc-rt-executor"
)
lli = os.path.join(config.llvm_tools_dir, "lli")
if config.host_os == "Darwin":
    orc_rt_path = "%s/liborc_rt_osx.a" % config.compiler_rt_libdir
else:
    orc_rt_path = "%s/liborc_rt%s.a" % (config.compiler_rt_libdir, config.target_suffix)

if config.libunwind_shared:
    config.available_features.add("libunwind-available")
    shared_libunwind_path = os.path.join(config.libunwind_install_dir, "libunwind.so")
    config.substitutions.append(("%shared_libunwind", shared_libunwind_path))


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(("%clang ", build_invocation([config.target_cflags])))
config.substitutions.append(
    ("%clangxx ", build_invocation(config.cxx_mode_flags + [config.target_cflags]))
)
config.substitutions.append(
    ("%clang_cl ", build_invocation(["--driver-mode=cl"] + [config.target_cflags]))
)
if config.host_os == "Windows":
    config.substitutions.append(
        (
            "%llvm_jitlink",
            (
                llvm_jitlink
                + " -orc-runtime="
                + orc_rt_path
                + " -no-process-syms=true -slab-allocate=64MB"
            ),
        )
    )
else:
    config.substitutions.append(
        ("%llvm_jitlink", (llvm_jitlink + " -orc-runtime=" + orc_rt_path))
    )
config.substitutions.append(
    ("%orc_rt_executor", orc_rt_executor_stem + "-" + config.host_arch)
)
config.substitutions.append(
    (
        "%lli_orc_jitlink",
        (lli + " -jit-kind=orc -jit-linker=jitlink -orc-runtime=" + orc_rt_path),
    )
)

# Default test suffixes.
config.suffixes = [".c", ".cpp", ".S", ".ll", ".test"]

# Exclude Inputs directories.
config.excludes = ["Inputs"]

if config.host_os not in ["Darwin", "FreeBSD", "Linux", "Windows"]:
    config.unsupported = True
