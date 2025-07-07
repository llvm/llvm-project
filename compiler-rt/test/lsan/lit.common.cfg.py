# -*- Python -*-

# Common configuration for running leak detection tests under LSan/ASan.

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

# Choose between standalone and LSan+(ASan|HWAsan) modes.
lsan_lit_test_mode = get_required_attr(config, "lsan_lit_test_mode")
target_arch = getattr(config, "target_arch", None)

if lsan_lit_test_mode == "Standalone":
    config.name = "LeakSanitizer-Standalone"
    lsan_cflags = ["-fsanitize=leak"]
    config.available_features.add("lsan-standalone")
elif lsan_lit_test_mode == "AddressSanitizer":
    config.name = "LeakSanitizer-AddressSanitizer"
    lsan_cflags = ["-fsanitize=address"]
    config.available_features.add("asan")
    if config.host_os == "NetBSD":
        config.substitutions.insert(0, ("%run", config.netbsd_noaslr_prefix))
elif lsan_lit_test_mode == "HWAddressSanitizer":
    config.name = "LeakSanitizer-HWAddressSanitizer"
    lsan_cflags = ["-fsanitize=hwaddress", "-fuse-ld=lld"]
    if target_arch == "x86_64":
        lsan_cflags = lsan_cflags + ["-fsanitize-hwaddress-experimental-aliasing"]
    config.available_features.add("hwasan")
    if config.host_os == "NetBSD":
        config.substitutions.insert(0, ("%run", config.netbsd_noaslr_prefix))
else:
    lit_config.fatal("Unknown LSan test mode: %r" % lsan_lit_test_mode)
config.name += config.name_suffix

# Platform-specific default LSAN_OPTIONS for lit tests.
default_common_opts_str = ":".join(list(config.default_sanitizer_opts))
default_lsan_opts = default_common_opts_str + ":detect_leaks=1"
if config.host_os == "Darwin":
    # On Darwin, we default to `abort_on_error=1`, which would make tests run
    # much slower. Let's override this and run lit tests with 'abort_on_error=0'.
    # Also, make sure we do not overwhelm the syslog while testing.
    default_lsan_opts += ":abort_on_error=0"
    default_lsan_opts += ":log_to_syslog=0"

if default_lsan_opts:
    config.environment["LSAN_OPTIONS"] = default_lsan_opts
    default_lsan_opts += ":"
config.substitutions.append(
    ("%env_lsan_opts=", "env LSAN_OPTIONS=" + default_lsan_opts)
)

if lit.util.which("strace"):
    config.available_features.add("strace")

clang_cflags = ["-O0", config.target_cflags] + config.debug_info_flags
if config.android:
    clang_cflags = clang_cflags + ["-fno-emulated-tls"]
clang_cxxflags = config.cxx_mode_flags + clang_cflags
lsan_incdir = config.test_source_root + "/../"
clang_lsan_cflags = clang_cflags + lsan_cflags + ["-I%s" % lsan_incdir]
clang_lsan_cxxflags = clang_cxxflags + lsan_cflags + ["-I%s" % lsan_incdir]

config.clang_cflags = clang_cflags
config.clang_cxxflags = clang_cxxflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(("%clang ", build_invocation(clang_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(clang_cxxflags)))
config.substitutions.append(("%clang_lsan ", build_invocation(clang_lsan_cflags)))
config.substitutions.append(("%clangxx_lsan ", build_invocation(clang_lsan_cxxflags)))
config.substitutions.append(("%clang_hwasan ", build_invocation(clang_lsan_cflags)))
config.substitutions.append(("%clangxx_hwasan ", build_invocation(clang_lsan_cxxflags)))


# LeakSanitizer tests are currently supported on
# Android{aarch64, x86, x86_64}, x86-64 Linux, PowerPC64 Linux, arm Linux, mips64 Linux, s390x Linux, loongarch64 Linux and x86_64 Darwin.
supported_android = (
    config.android
    and config.target_arch in ["x86_64", "i386", "aarch64"]
    and "android-thread-properties-api" in config.available_features
)
supported_linux = (
    (not config.android)
    and config.host_os == "Linux"
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
supported_darwin = config.host_os == "Darwin" and config.target_arch in ["x86_64"]
supported_netbsd = config.host_os == "NetBSD" and config.target_arch in [
    "x86_64",
    "i386",
]
if not (supported_android or supported_linux or supported_darwin or supported_netbsd):
    config.unsupported = True

# Don't support Thumb due to broken fast unwinder
if re.search("mthumb", config.target_cflags) is not None:
    config.unsupported = True

# HWASAN tests require lld because without D65857, ld.bfd and ld.gold would
# generate a corrupted binary. Mark them unsupported if lld is not available.
if "hwasan" in config.available_features and not config.has_lld:
    config.unsupported = True

config.suffixes = [".c", ".cpp", ".mm"]
