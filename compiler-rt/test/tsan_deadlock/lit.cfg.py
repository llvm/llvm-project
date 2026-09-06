# -*- Python -*-

import os

# Setup config name.
config.name = "DeadlockSanitizer" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

default_tsan_deadlock_opts = ""

if config.target_os == "Darwin":
    default_tsan_deadlock_opts += "abort_on_error=0"

if default_tsan_deadlock_opts:
    config.environment["TSAN_DEADLOCK_OPTIONS"] = default_tsan_deadlock_opts
    default_tsan_deadlock_opts += ":"
config.substitutions.append(
    (
        "%env_tsan_deadlock_opts=",
        "env TSAN_DEADLOCK_OPTIONS=" + default_tsan_deadlock_opts,
    )
)

clang_tsan_deadlock_cflags = (
    ["-fsanitize=thread-deadlock", "-Wall"]
    + [config.target_cflags]
    + config.debug_info_flags
)
clang_tsan_deadlock_cxxflags = (
    config.cxx_mode_flags + clang_tsan_deadlock_cflags + ["-std=c++11"]
)


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(
    ("%clang_tsan_deadlock ", build_invocation(clang_tsan_deadlock_cflags))
)
config.substitutions.append(
    ("%clangxx_tsan_deadlock ", build_invocation(clang_tsan_deadlock_cxxflags))
)

config.suffixes = [".c", ".cpp"]

if config.target_os not in ["FreeBSD", "Linux", "Darwin", "NetBSD"]:
    config.unsupported = True
