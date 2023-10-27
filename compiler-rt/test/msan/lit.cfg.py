# -*- Python -*-

import os

# Setup config name.
config.name = "MemorySanitizer" + getattr(config, "name_suffix", "default")

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup default compiler flags used with -fsanitize=memory option.
clang_msan_cflags = (
    [
        "-fsanitize=memory",
        "-mno-omit-leaf-frame-pointer",
        "-fno-omit-frame-pointer",
        "-fno-optimize-sibling-calls",
    ]
    + [config.target_cflags]
    + config.debug_info_flags
)
# Some Msan tests leverage backtrace() which requires libexecinfo on FreeBSD.
if config.host_os == "FreeBSD":
    clang_msan_cflags += ["-lexecinfo", "-fPIC"]
# On SystemZ we need -mbackchain to make the fast unwinder work.
if config.target_arch == "s390x":
    clang_msan_cflags.append("-mbackchain")
clang_msan_cxxflags = config.cxx_mode_flags + clang_msan_cflags

# Flags for KMSAN invocation. This is C-only, we're not interested in C++.
clang_kmsan_cflags = (
    ["-fsanitize=kernel-memory"] + [config.target_cflags] + config.debug_info_flags
)


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(("%clang_msan ", build_invocation(clang_msan_cflags)))
config.substitutions.append(("%clangxx_msan ", build_invocation(clang_msan_cxxflags)))
config.substitutions.append(("%clang_kmsan ", build_invocation(clang_kmsan_cflags)))

# Default test suffixes.
config.suffixes = [".c", ".cpp"]

if config.host_os not in ["Linux", "NetBSD", "FreeBSD"]:
    config.unsupported = True

# For mips64, mips64el we have forced store_context_size to 1 because these
# archs use slow unwinder which is not async signal safe. Therefore we only
# check the first frame since store_context size is 1.
if config.host_arch in ["mips64", "mips64el"]:
    config.substitutions.append(("CHECK-%short-stack", "CHECK-SHORT-STACK"))
else:
    config.substitutions.append(("CHECK-%short-stack", "CHECK-FULL-STACK"))

if config.host_os == "NetBSD":
    config.substitutions.insert(0, ("%run", config.netbsd_noaslr_prefix))
