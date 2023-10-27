# -*- Python -*-

import os

# Setup config name.
config.name = "XRay" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup default compiler flags use with -fxray-instrument option.
clang_xray_cflags = ["-fxray-instrument", config.target_cflags]

# If libc++ was used to build XRAY libraries, libc++ is needed. Fix applied
# to Linux only since -rpath may not be portable. This can be extended to
# other platforms.
if config.libcxx_used == "1" and config.host_os == "Linux":
    clang_xray_cflags = clang_xray_cflags + (
        ["-L%s -lc++ -Wl,-rpath=%s" % (config.llvm_shlib_dir, config.llvm_shlib_dir)]
    )

clang_xray_cxxflags = config.cxx_mode_flags + clang_xray_cflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


# Assume that llvm-xray is in the config.llvm_tools_dir.
llvm_xray = os.path.join(config.llvm_tools_dir, "llvm-xray")

# Setup substitutions.
if config.host_os == "Linux":
    libdl_flag = "-ldl"
else:
    libdl_flag = ""

config.substitutions.append(("%clang ", build_invocation([config.target_cflags])))
config.substitutions.append(
    ("%clangxx ", build_invocation(config.cxx_mode_flags + [config.target_cflags]))
)
config.substitutions.append(("%clang_xray ", build_invocation(clang_xray_cflags)))
config.substitutions.append(("%clangxx_xray", build_invocation(clang_xray_cxxflags)))
config.substitutions.append(("%llvm_xray", llvm_xray))
config.substitutions.append(
    (
        "%xraylib",
        (
            "-lm -lpthread %s -lrt -L%s "
            "-Wl,-whole-archive -lclang_rt.xray%s -Wl,-no-whole-archive"
        )
        % (libdl_flag, config.compiler_rt_libdir, config.target_suffix),
    )
)

# Default test suffixes.
config.suffixes = [".c", ".cpp"]

if config.host_os not in ["FreeBSD", "Linux", "NetBSD", "OpenBSD"]:
    config.unsupported = True
elif "64" not in config.host_arch:
    if "arm" in config.host_arch:
        if "-mthumb" in config.target_cflags:
            config.unsupported = True
    else:
        config.unsupported = True

if config.host_os == "NetBSD":
    config.substitutions.insert(0, ("%run", config.netbsd_nomprotect_prefix))
