# -*- Python -*-

import os


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
config.name = "InstrumentorTools-" + config.target_arch

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup executable root.
if (
    hasattr(config, "instrumentor_lit_binary_dir")
    and config.instrumentor_lit_binary_dir is not None
):
    config.test_exec_root = os.path.join(
        config.instrumentor_lit_binary_dir, config.name
    )

# Test suffixes.
config.suffixes = [".c", ".cpp", ".m", ".mm", ".ll", ".test"]

# What to exclude.
config.excludes = ["Inputs"]

# Clang flags.
target_cflags = [get_required_attr(config, "target_cflags")]
clang_cflags = target_cflags
clang_cxxflags = config.cxx_mode_flags + clang_cflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


def make_lib_name(name):
    if config.target_os != "Darwin":
        return "clang_rt.instrumentor_" + name
    return "clang_rt.instrumentor_" + name + "_osx"


def make_lib_substitutions(name):
    config.substitutions.append(("%" + name + "_lib", make_lib_name(name)))


# Add clang substitutions.
config.substitutions.append(("%clang ", build_invocation(clang_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(clang_cxxflags)))

tools = ["flop_counter", "fp_precision_analysis"]
for tool in tools:
    make_lib_substitutions(tool)

config.substitutions.append(("%lib_dir", config.compiler_rt_libdir))

# Add path to Pointer Tracking runtime library
config.substitutions.append(("%pointer_tracking_lib_dir", config.compiler_rt_libdir))

pointer_tracking_lib = make_lib_name("pointer_tracking")
config.substitutions.append(("%pointer_tracking_lib", pointer_tracking_lib))

# Add path to instrumentor config files
config_dir = os.path.join(
    config.test_source_root, "..", "..", "lib", "instrumentor-tools"
)
config.substitutions.append(("%config_dir", config_dir))

# Check if running on a supported platform
if config.target_os not in [
    "Darwin",
    "Linux",
    "FreeBSD",
]:
    config.unsupported = True
