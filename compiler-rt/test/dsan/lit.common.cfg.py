# -*- Python -*-

# Common configuration for running double-free detection tests under DSan.

import os
import re

import lit.util


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal(
            "No attribute %r in test configuration! You may need to run "
            "tests from your build directory or add this attribute "
            "to lit.site.cfg.py " % attr_name
        )
    return attr_value


# Setup source root.
config.test_source_root = os.path.dirname(__file__)

dsan_lit_test_mode = get_required_attr(config, "dsan_lit_test_mode")

if dsan_lit_test_mode == "Standalone":
    config.name = "DoubleFreeSanitizer-Standalone"
    dsan_cflags = ["-fsanitize=doublefree"]
    config.available_features.add("dsan-standalone")
else:
    lit_config.fatal("Unknown DSan test mode: %r" % dsan_lit_test_mode)
config.name += config.name_suffix

# Platform-specific default DSAN_OPTIONS for lit tests.
default_common_opts_str = ":".join(list(config.default_sanitizer_opts))
default_dsan_opts = default_common_opts_str
if config.target_os == "Darwin":
    # On Darwin, we default to `abort_on_error=1`, which would make tests run
    # much slower. Let's override this and run lit tests with 'abort_on_error=0'.
    # Also, make sure we do not overwhelm the syslog while testing.
    default_dsan_opts += ":abort_on_error=0"
    default_dsan_opts += ":log_to_syslog=0"

if default_dsan_opts:
    config.environment["DSAN_OPTIONS"] = default_dsan_opts
    default_dsan_opts += ":"
config.substitutions.append(
    ("%env_dsan_opts=", "env DSAN_OPTIONS=" + default_dsan_opts)
)

if lit.util.which("strace"):
    config.available_features.add("strace")

clang_cflags = ["-O0", config.target_cflags] + config.debug_info_flags
if config.android:
    clang_cflags = clang_cflags + ["-fno-emulated-tls"]
clang_cxxflags = config.cxx_mode_flags + clang_cflags
dsan_incdir = config.test_source_root + "/../"
clang_dsan_cflags = clang_cflags + dsan_cflags + ["-I%s" % dsan_incdir]
clang_dsan_cxxflags = clang_cxxflags + dsan_cflags + ["-I%s" % dsan_incdir]

config.clang_cflags = clang_cflags
config.clang_cxxflags = clang_cxxflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(("%clang ", build_invocation(clang_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(clang_cxxflags)))
config.substitutions.append(("%clang_dsan ", build_invocation(clang_dsan_cflags)))
config.substitutions.append(("%clangxx_dsan ", build_invocation(clang_dsan_cxxflags)))


# DoubleFreeSanitizer tests are currently supported on
# Android{aarch64, x86, x86_64}, x86-64 Linux, PowerPC64 Linux, arm Linux, mips64 Linux, s390x Linux, loongarch64 Linux and x86_64 Darwin.
supported_android = (
    config.android
    and config.target_arch in ["x86_64", "i386", "aarch64"]
    and "android-thread-properties-api" in config.available_features
)
supported_linux = (
    (not config.android)
    and config.target_os == "Linux"
    and config.host_arch
    in [
        "aarch64",
        "x86_64",
        "ppc64",
        "ppc64le",
        "mips64",
        "riscv64",
        "arm",
        "armhf",
        "armv7l",
        "s390x",
        "loongarch64",
    ]
)
supported_darwin = config.target_os == "Darwin" and config.target_arch in ["x86_64"]
supported_netbsd = config.target_os == "NetBSD" and config.target_arch in [
    "x86_64",
    "i386",
]
if not (supported_android or supported_linux or supported_darwin or supported_netbsd):
    config.unsupported = True

# Don't support Thumb due to broken fast unwinder
if re.search("mthumb", config.target_cflags) is not None:
    config.unsupported = True

config.suffixes = [".c", ".cpp", ".mm"]
