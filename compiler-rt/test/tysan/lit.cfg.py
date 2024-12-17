# -*- Python -*-

import os
import platform
import re
import shlex

import lit.formats


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is not None:
        return attr_value

    lit_config.fatal(
        "No attribute %r in test configuration! You may need to run "
        "tests from your build directory or add this attribute "
        "to lit.site.cfg.py " % attr_name
    )


def push_dynamic_library_lookup_path(config, new_path):
    if platform.system() == "Windows":
        dynamic_library_lookup_var = "PATH"
    elif platform.system() == "Darwin":
        dynamic_library_lookup_var = "DYLD_LIBRARY_PATH"
    else:
        dynamic_library_lookup_var = "LD_LIBRARY_PATH"

    new_ld_library_path = os.path.pathsep.join(
        (new_path, config.environment.get(dynamic_library_lookup_var, ""))
    )
    config.environment[dynamic_library_lookup_var] = new_ld_library_path

    if platform.system() == "FreeBSD":
        dynamic_library_lookup_var = "LD_32_LIBRARY_PATH"
        new_ld_32_library_path = os.path.pathsep.join(
            (new_path, config.environment.get(dynamic_library_lookup_var, ""))
        )
        config.environment[dynamic_library_lookup_var] = new_ld_32_library_path

    if platform.system() == "SunOS":
        dynamic_library_lookup_var = "LD_LIBRARY_PATH_32"
        new_ld_library_path_32 = os.path.pathsep.join(
            (new_path, config.environment.get(dynamic_library_lookup_var, ""))
        )
        config.environment[dynamic_library_lookup_var] = new_ld_library_path_32

        dynamic_library_lookup_var = "LD_LIBRARY_PATH_64"
        new_ld_library_path_64 = os.path.pathsep.join(
            (new_path, config.environment.get(dynamic_library_lookup_var, ""))
        )
        config.environment[dynamic_library_lookup_var] = new_ld_library_path_64


# Setup config name.
config.name = "TypeSanitizer" + config.name_suffix

# Platform-specific default TYSAN_OPTIONS for lit tests.
default_tysan_opts = list(config.default_sanitizer_opts)

default_tysan_opts_str = ":".join(default_tysan_opts)
if default_tysan_opts_str:
    config.environment["TYSAN_OPTIONS"] = default_tysan_opts_str
    default_tysan_opts_str += ":"
config.substitutions.append(
    ("%env_tysan_opts=", "env TYSAN_OPTIONS=" + default_tysan_opts_str)
)

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

if config.host_os not in ["FreeBSD", "NetBSD"]:
    libdl_flag = "-ldl"
else:
    libdl_flag = ""

# GCC-ASan doesn't link in all the necessary libraries automatically, so
# we have to do it ourselves.
if config.compiler_id == "GNU":
    extra_link_flags = ["-pthread", "-lstdc++", libdl_flag]
else:
    extra_link_flags = []

# Setup default compiler flags used with -fsanitize=address option.
# FIXME: Review the set of required flags and check if it can be reduced.
target_cflags = [get_required_attr(config, "target_cflags")] + extra_link_flags
target_cxxflags = config.cxx_mode_flags + target_cflags
clang_tysan_static_cflags = (
    [
        "-fsanitize=type",
        "-mno-omit-leaf-frame-pointer",
        "-fno-omit-frame-pointer",
        "-fno-optimize-sibling-calls",
    ]
    + config.debug_info_flags
    + target_cflags
)
if config.target_arch == "s390x":
    clang_tysan_static_cflags.append("-mbackchain")
clang_tysan_static_cxxflags = config.cxx_mode_flags + clang_tysan_static_cflags

clang_tysan_cflags = clang_tysan_static_cflags
clang_tysan_cxxflags = clang_tysan_static_cxxflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(("%clang ", build_invocation(target_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(target_cxxflags)))
config.substitutions.append(("%clang_tysan ", build_invocation(clang_tysan_cflags)))
config.substitutions.append(("%clangxx_tysan ", build_invocation(clang_tysan_cxxflags)))


# FIXME: De-hardcode this path.
tysan_source_dir = os.path.join(
    get_required_attr(config, "compiler_rt_src_root"), "lib", "tysan"
)
python_exec = shlex.quote(get_required_attr(config, "python_executable"))

# Set LD_LIBRARY_PATH to pick dynamic runtime up properly.
push_dynamic_library_lookup_path(config, config.compiler_rt_libdir)

# Default test suffixes.
config.suffixes = [".c", ".cpp"]

if config.host_os == "Darwin":
    config.suffixes.append(".mm")

if config.host_os == "Windows":
    config.substitutions.append(("%fPIC", ""))
    config.substitutions.append(("%fPIE", ""))
    config.substitutions.append(("%pie", ""))
else:
    config.substitutions.append(("%fPIC", "-fPIC"))
    config.substitutions.append(("%fPIE", "-fPIE"))
    config.substitutions.append(("%pie", "-pie"))

# Only run the tests on supported OSs.
if config.host_os not in [
    "Linux",
    "Darwin",
]:
    config.unsupported = Tr
