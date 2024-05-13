# -*- Python -*-

import os

# Setup config name.
config.name = "GWP-ASan" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = [".c", ".cpp", ".test"]

# C & CXX flags.
c_flags = [config.target_cflags]

cxx_flags = c_flags + config.cxx_mode_flags + ["-std=c++14"]

libscudo_standalone = os.path.join(
    config.compiler_rt_libdir, "libclang_rt.scudo_standalone%s.a" % config.target_suffix
)
libscudo_standalone_cxx = os.path.join(
    config.compiler_rt_libdir,
    "libclang_rt.scudo_standalone_cxx%s.a" % config.target_suffix,
)

scudo_link_flags = [
    "-pthread",
    "-Wl,--whole-archive",
    libscudo_standalone,
    "-Wl,--no-whole-archive",
]
scudo_link_cxx_flags = [
    "-Wl,--whole-archive",
    libscudo_standalone_cxx,
    "-Wl,--no-whole-archive",
]

# -rdynamic is necessary for online function symbolization.
gwp_asan_flags = ["-rdynamic"] + scudo_link_flags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


# Add substitutions.
config.substitutions.append(("%clang ", build_invocation(c_flags)))
config.substitutions.append(
    ("%clang_gwp_asan ", build_invocation(c_flags + gwp_asan_flags))
)
config.substitutions.append(
    (
        "%clangxx_gwp_asan ",
        build_invocation(cxx_flags + gwp_asan_flags + scudo_link_cxx_flags),
    )
)

# Platform-specific default GWP_ASAN for lit tests. Ensure that GWP-ASan is
# enabled and that it samples every allocation.
default_gwp_asan_options = "GWP_ASAN_Enabled=1:GWP_ASAN_SampleRate=1"

config.environment["SCUDO_OPTIONS"] = default_gwp_asan_options
default_gwp_asan_options += ":"
config.substitutions.append(
    ("%env_scudo_options=", "env SCUDO_OPTIONS=" + default_gwp_asan_options)
)

# GWP-ASan tests are currently supported on Linux only.
if config.host_os not in ["Linux"]:
    config.unsupported = True
