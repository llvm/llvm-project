# -*- Python -*-

import os
import platform

config.name = "NASan" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = [".c", ".cpp", ".test"]

# C & CXX flags.
c_flags = [config.target_cflags]

# CXX flags
cxx_flags = c_flags + config.cxx_mode_flags + ["-std=c++17"]

nasan_flags = [
    "-fsanitize=noalias",
    "-g",
    "-mno-omit-leaf-frame-pointer",
    "-fno-omit-frame-pointer",
]


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


# Add substitutions.
config.substitutions.append(("%clang ", build_invocation(c_flags)))
config.substitutions.append(("%clang_nasan ", build_invocation(c_flags + nasan_flags)))
config.substitutions.append(
    ("%clangxx_nasan ", build_invocation(cxx_flags + nasan_flags))
)

# NASan tests are currently supported on macOS and Linux.
if config.target_os not in ["Darwin", "Linux"]:
    config.unsupported = True

# x86_64h requires Haswell hardware and can't run on ARM64 Macs
if config.target_os == "Darwin" and platform.machine() == "arm64":
    if config.target_arch == "x86_64h":
        config.unsupported = True
