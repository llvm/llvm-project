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


# Define %clang and %clangxx substitutions to use in test RUN lines.
config.substitutions.append(
    ("%clang ", " " + config.clang + " " + " ".join(extra_flags) + " ")
)

if config.host_os == "Darwin":
    config.substitutions.append(
        ("%macos_version_major", str(config.darwin_osx_version[0]))
    )
    config.substitutions.append(
        ("%macos_version_minor", str(config.darwin_osx_version[1]))
    )
    config.substitutions.append(
        ("%macos_version_subminor", str(config.darwin_osx_version[2]))
    )
