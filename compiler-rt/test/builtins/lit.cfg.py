# -*- Python -*-

import os

# Setup config name.
config.name = "Builtins"

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = [".c", ".cpp", ".m", ".mm", ".test"]
extra_flags = ["-Wall"]
if config.compiler_id == "GNU":
    # detect incorrect declarations of libgcc functions
    extra_flags.append("-Werror=builtin-declaration-mismatch")

# On Windows, when testing i386 architecture, explicitly pass the target triple
# to ensure clang compiles for i386 instead of defaulting to x64.
if config.target_os == "Windows" and config.target_arch == "i386":
    extra_flags.append("--target=i386-pc-windows-msvc")


# Define %clang and %clangxx substitutions to use in test RUN lines.
config.substitutions.append(
    ("%clang ", " " + config.clang + " " + " ".join(extra_flags) + " ")
)

if config.target_os == "Darwin":
    config.substitutions.append(
        ("%macos_version_major", str(config.darwin_os_version[0]))
    )
    config.substitutions.append(
        ("%macos_version_minor", str(config.darwin_os_version[1]))
    )
    config.substitutions.append(
        ("%macos_version_subminor", str(config.darwin_os_version[2]))
    )
