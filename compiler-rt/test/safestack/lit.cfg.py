# -*- Python -*-

import os

# Setup config name.
config.name = "SafeStack"

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = [".c", ".cpp", ".m", ".mm", ".ll", ".test"]

# Add clang substitutions.
config.substitutions.append(
    ("%clang_nosafestack ", config.clang + " -O0 -fno-sanitize=safe-stack ")
)
config.substitutions.append(
    ("%clang_safestack ", config.clang + " -O0 -fsanitize=safe-stack ")
)

if config.lto_supported:
    config.substitutions.append(
        (
            r"%clang_lto_safestack ",
            " ".join([config.clang] + config.lto_flags + ["-fsanitize=safe-stack "]),
        )
    )

if config.host_os not in ["Linux", "FreeBSD", "NetBSD"]:
    config.unsupported = True
