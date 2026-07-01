# -*- Python -*-

import glob
import os
import re
import subprocess


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal(
            "No attribute %r in test configuration! You may need to run "
            "tests from your build directory or add this attribute "
            "to lit.site.cfg.py " % attr_name
        )
    return attr_value


# Setup config name.
config.name = "Profile-" + config.target_arch

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup executable root.
if (
    hasattr(config, "profile_lit_binary_dir")
    and config.profile_lit_binary_dir is not None
):
    config.test_exec_root = os.path.join(config.profile_lit_binary_dir, config.name)

target_is_msvc = bool(re.match(r".*-windows-msvc$", config.target_triple))

if config.target_os in ["Linux"]:
    extra_link_flags = ["-ldl"]
elif target_is_msvc:
    # InstrProf is incompatible with incremental linking. Disable it as a
    # workaround.
    extra_link_flags = ["-Wl,-incremental:no"]
else:
    extra_link_flags = []

# Test suffixes.
config.suffixes = [".c", ".cpp", ".m", ".mm", ".ll", ".test"]

# What to exclude.
config.excludes = ["Inputs"]

# Clang flags.
target_cflags = [get_required_attr(config, "target_cflags")]
clang_cflags = target_cflags + extra_link_flags
clang_cxxflags = config.cxx_mode_flags + clang_cflags

# TODO: target_cflags can sometimes contain C++ only flags like -stdlib=<FOO>, which are
#       ignored when compiling as C code. Passing this flag when compiling as C results in
#       warnings that break tests that use -Werror.
#       We remove -stdlib= from the cflags here to avoid problems, but the interaction between
#       CMake and compiler-rt's tests should be reworked so that cflags don't contain C++ only
#       flags.
clang_cflags = [
    flag.replace("-stdlib=libc++", "").replace("-stdlib=libstdc++", "")
    for flag in clang_cflags
]


def build_invocation(compile_flags, with_lto=False):
    lto_flags = []
    if with_lto and config.lto_supported:
        lto_flags += config.lto_flags
    return " " + " ".join([config.clang] + lto_flags + compile_flags) + " "


def exclude_unsupported_files_for_aix(dirname):
    for filename in os.listdir(dirname):
        source_path = os.path.join(dirname, filename)
        if os.path.isdir(source_path):
            continue
        f = open(source_path, "r")
        try:
            data = f.read()
            # rpath is not supported on AIX, exclude all tests with them.
            if ( "-rpath" in data ):
                config.excludes += [filename]
        finally:
            f.close()


# Add clang substitutions.
config.substitutions.append(("%clang ", build_invocation(clang_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(clang_cxxflags)))

config.substitutions.append(
    ("%clang_profgen ", build_invocation(clang_cflags) + " -fprofile-instr-generate ")
)
config.substitutions.append(
    ("%clang_profgen=", build_invocation(clang_cflags) + " -fprofile-instr-generate=")
)
config.substitutions.append(
    (
        "%clangxx_profgen ",
        build_invocation(clang_cxxflags) + " -fprofile-instr-generate ",
    )
)
config.substitutions.append(
    (
        "%clangxx_profgen=",
        build_invocation(clang_cxxflags) + " -fprofile-instr-generate=",
    )
)

config.substitutions.append(
    ("%clang_pgogen ", build_invocation(clang_cflags) + " -fprofile-generate ")
)
config.substitutions.append(
    ("%clang_pgogen=", build_invocation(clang_cflags) + " -fprofile-generate=")
)
config.substitutions.append(
    ("%clangxx_pgogen ", build_invocation(clang_cxxflags) + " -fprofile-generate ")
)
config.substitutions.append(
    ("%clangxx_pgogen=", build_invocation(clang_cxxflags) + " -fprofile-generate=")
)

config.substitutions.append(
    ("%clang_cspgogen ", build_invocation(clang_cflags) + " -fcs-profile-generate ")
)
config.substitutions.append(
    ("%clang_cspgogen=", build_invocation(clang_cflags) + " -fcs-profile-generate=")
)
config.substitutions.append(
    ("%clangxx_cspgogen ", build_invocation(clang_cxxflags) + " -fcs-profile-generate ")
)
config.substitutions.append(
    ("%clangxx_cspgogen=", build_invocation(clang_cxxflags) + " -fcs-profile-generate=")
)

config.substitutions.append(
    ("%clang_profuse=", build_invocation(clang_cflags) + " -fprofile-instr-use=")
)
config.substitutions.append(
    ("%clangxx_profuse=", build_invocation(clang_cxxflags) + " -fprofile-instr-use=")
)

config.substitutions.append(
    ("%clang_pgouse=", build_invocation(clang_cflags) + " -fprofile-use=")
)
config.substitutions.append(
    ("%clangxx_profuse=", build_invocation(clang_cxxflags) + " -fprofile-instr-use=")
)

config.substitutions.append(
    (
        "%clang_lto_profgen=",
        build_invocation(clang_cflags, True) + " -fprofile-instr-generate=",
    )
)

if config.target_os not in [
    "Windows",
    "Darwin",
    "FreeBSD",
    "Linux",
    "NetBSD",
    "SunOS",
    "AIX",
    "Haiku",
]:
    config.unsupported = True

config.substitutions.append(
    ("%shared_lib_flag", "-dynamiclib" if (config.target_os == "Darwin") else "-shared")
)

if config.target_os in ["AIX"]:
    config.available_features.add("system-aix")
    exclude_unsupported_files_for_aix(config.test_source_root)
    exclude_unsupported_files_for_aix(config.test_source_root + "/Posix")

if config.target_arch in ["armv7l"]:
    config.unsupported = True

if config.android:
    config.unsupported = True

if config.have_curl:
    config.available_features.add("curl")

if config.target_os in ("AIX", "Darwin", "Linux"):
    config.available_features.add("continuous-mode")

# GPU (HIP/AMDGPU) device-profile tests.
#
# The GPU/ and AMDGPU/ subdirectories exercise the device-PGO drain end to end
# and need a real AMD GPU plus a ROCm/HIP install. Detect that here and, when
# present, expose the features ('hip', 'amdgpu', 'multi-device') and
# substitutions ('%amdgpu_arch', '%hip_lib_path') those tests use. Without a GPU
# the subdirectory lit.local.cfg.py files mark themselves unsupported, so the
# tests report UNSUPPORTED instead of failing.
#
# Both knobs are overridable from the command line, e.g.:
#   llvm-lit --param amdgpu_arch=gfx90a --param hip_lib_path=/opt/rocm/lib ...
config.suffixes.append(".hip")


def _amdgpu_archs():
    # config.clang is a wrapped command ("<wrappers> <clang>"); the clang path is
    # the last token. amdgpu-arch ships next to it and prints one line per GPU.
    clang_path = config.clang.split()[-1] if config.clang else ""
    tool = os.path.join(os.path.dirname(clang_path), "amdgpu-arch")
    if not os.path.exists(tool):
        return []
    try:
        proc = subprocess.run(tool, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _hip_lib_path():
    # An explicit --param hip_lib_path=DIR is authoritative; otherwise probe the
    # usual ROCm locations for libamdhip64.
    explicit = lit_config.params.get("hip_lib_path")
    if explicit:
        candidates = [explicit]
    else:
        candidates = []
        for var in ("ROCM_PATH", "HIP_PATH"):
            if os.environ.get(var):
                candidates.append(os.path.join(os.environ[var], "lib"))
        candidates.append("/opt/rocm/lib")
    for directory in candidates:
        if directory and glob.glob(os.path.join(directory, "libamdhip64.so*")):
            return directory
    return None


_amdgpu_arch_list = _amdgpu_archs()
_hip_lib_dir = _hip_lib_path()
if _amdgpu_arch_list and _hip_lib_dir:
    config.available_features.add("hip")
    config.available_features.add("amdgpu")
    if len(_amdgpu_arch_list) >= 2:
        config.available_features.add("multi-device")
    config.substitutions.append(
        ("%amdgpu_arch", lit_config.params.get("amdgpu_arch", "native"))
    )
    config.substitutions.append(("%hip_lib_path", _hip_lib_dir))
    # The GPU tests share the device(s) and pin HIP_VISIBLE_DEVICES, so they must
    # not run concurrently with each other. The subdirectories opt into this
    # group; the size-1 cap serializes them while leaving the CPU profile tests
    # fully parallel.
    lit_config.parallelism_groups["gpu"] = 1
