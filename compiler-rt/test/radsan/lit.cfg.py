# -*- Python -*-

import os

# Setup config name.
config.name = "RADSAN" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup default compiler flags use with -fradsan-instrument option.
clang_radsan_cflags = ["-fradsan-instrument", config.target_cflags]

# If libc++ was used to build radsan libraries, libc++ is needed. Fix applied
# to Linux only since -rpath may not be portable. This can be extended to
# other platforms.
if config.libcxx_used == "1" and config.host_os == "Linux":
    clang_radsan_cflags = clang_radsan_cflags + (
        ["-L%s -lc++ -Wl,-rpath=%s" % (config.llvm_shlib_dir, config.llvm_shlib_dir)]
    )

clang_radsan_cxxflags = config.cxx_mode_flags + clang_radsan_cflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


# Assume that llvm-radsan is in the config.llvm_tools_dir.
llvm_radsan = os.path.join(config.llvm_tools_dir, "llvm-radsan")

# Setup substitutions.
if config.host_os == "Linux":
    libdl_flag = "-ldl"
else:
    libdl_flag = ""

config.substitutions.append(("%clang ", build_invocation([config.target_cflags])))
config.substitutions.append(
    ("%clangxx ", build_invocation(config.cxx_mode_flags + [config.target_cflags]))
)
config.substitutions.append(("%clang_radsan ", build_invocation(clang_radsan_cflags)))
config.substitutions.append(("%clangxx_radsan", build_invocation(clang_radsan_cxxflags)))
config.substitutions.append(("%llvm_radsan", llvm_radsan))
config.substitutions.append(
    (
        "%radsanlib",
        (
            "-lm -lpthread %s -lrt -L%s "
            "-Wl,-whole-archive -lclang_rt.radsan%s -Wl,-no-whole-archive"
        )
        % (libdl_flag, config.compiler_rt_libdir, config.target_suffix),
    )
)

# Default test suffixes.
config.suffixes = [".c", ".cpp"]

if config.host_os not in ["Darwin", "FreeBSD", "Linux", "NetBSD", "OpenBSD"]:
    config.unsupported = True
elif "64" not in config.host_arch:
    if "arm" in config.host_arch:
        if "-mthumb" in config.target_cflags:
            config.unsupported = True
    else:
        config.unsupported = True

if config.host_os == "NetBSD":
    config.substitutions.insert(0, ("%run", config.netbsd_nomprotect_prefix))
