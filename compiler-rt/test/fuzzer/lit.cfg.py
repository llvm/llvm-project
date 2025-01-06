import lit.formats
import sys
import os

config.name = "libFuzzer" + config.name_suffix
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)
config.available_features.add(config.target_arch)

# Choose between lit's internal shell pipeline runner and a real shell.  If
# LIT_USE_INTERNAL_SHELL is in the environment, we use that as an override.
use_lit_shell = os.environ.get("LIT_USE_INTERNAL_SHELL")
if use_lit_shell:
    # 0 is external, "" is default, and everything else is internal.
    execute_external = use_lit_shell == "0"
else:
    # Otherwise we default to internal on Windows and external elsewhere, as
    # bash on Windows is usually very slow.
    execute_external = not sys.platform in ["win32"]

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(execute_external)

# LeakSanitizer is not supported on OSX or Windows right now.
if (
    sys.platform.startswith("darwin")
    or sys.platform.startswith("freebsd")
    or sys.platform.startswith("win")
):
    lit_config.note("lsan feature unavailable")
else:
    config.available_features.add("lsan")

# MemorySanitizer is not supported on OSX or Windows right now
if (
    sys.platform.startswith("darwin")
    or sys.platform.startswith("win")
):
    lit_config.note("msan feature unavailable")
    assert "msan" not in config.available_features
elif config.target_arch == "i386":
    assert "msan" not in config.available_features
else:
    config.available_features.add("msan")

if sys.platform.startswith("win") or sys.platform.startswith("cygwin"):
    config.available_features.add("windows")

if sys.platform.startswith("darwin"):
    config.available_features.add("darwin")

if sys.platform.startswith("linux"):
    # Note the value of ``sys.platform`` is not consistent
    # between python 2 and 3, hence the use of ``.startswith()``.
    config.available_features.add("linux")

if config.arm_thumb:
    config.available_features.add("thumb")

config.substitutions.append(("%build_dir", config.cmake_binary_dir))
libfuzzer_src_root = os.path.join(config.compiler_rt_src_root, "lib", "fuzzer")
config.substitutions.append(("%libfuzzer_src", libfuzzer_src_root))

config.substitutions.append(("%python", '"%s"' % (sys.executable)))


def generate_compiler_cmd(is_cpp=True, fuzzer_enabled=True, msan_enabled=False):
    compiler_cmd = config.clang
    extra_cmd = config.target_flags

    if is_cpp:
        std_cmd = "--driver-mode=g++"
    else:
        std_cmd = ""

    if msan_enabled:
        sanitizers = ["memory"]
    else:
        sanitizers = ["address"]
    if fuzzer_enabled:
        sanitizers.append("fuzzer")
    sanitizers_cmd = "-fsanitize=%s" % ",".join(sanitizers)

    # Since clang_rt.fuzzer-x86_64.lib is built without -fsanitize=address and
    # we're building with that flag here, the MSVC STL needs all libs to be
    # compiled with the same -fsanitize flags, see:
    # https://learn.microsoft.com/en-us/cpp/sanitizers/error-container-overflow?view=msvc-170
    # Avoids linker error:
    #   /failifmismatch: mismatch detected for 'annotate_string'
    if "windows" in config.available_features:
        extra_cmd = extra_cmd + " -D_DISABLE_VECTOR_ANNOTATION -D_DISABLE_STRING_ANNOTATION"

    if "darwin" in config.available_features and getattr(
        config, "darwin_linker_version", None
    ):
        extra_cmd = extra_cmd + " -mlinker-version=" + config.darwin_linker_version

    return " ".join(
        [
            compiler_cmd,
            config.target_cflags,
            std_cmd,
            "-O2 -gline-tables-only",
            sanitizers_cmd,
            "-I%s" % libfuzzer_src_root,
            extra_cmd,
        ]
    )


config.substitutions.append(
    ("%cpp_compiler", generate_compiler_cmd(is_cpp=True, fuzzer_enabled=True))
)

config.substitutions.append(
    ("%c_compiler", generate_compiler_cmd(is_cpp=False, fuzzer_enabled=True))
)

config.substitutions.append(
    (
        "%no_fuzzer_cpp_compiler",
        generate_compiler_cmd(is_cpp=True, fuzzer_enabled=False),
    )
)

config.substitutions.append(
    ("%no_fuzzer_c_compiler", generate_compiler_cmd(is_cpp=False, fuzzer_enabled=False))
)

config.substitutions.append(
    (
        "%msan_compiler",
        generate_compiler_cmd(is_cpp=True, fuzzer_enabled=True, msan_enabled=True),
    )
)

default_asan_opts_str = ":".join(config.default_sanitizer_opts)
if default_asan_opts_str:
    config.environment["ASAN_OPTIONS"] = default_asan_opts_str
    default_asan_opts_str += ":"
config.substitutions.append(
    ("%env_asan_opts=", "env ASAN_OPTIONS=" + default_asan_opts_str)
)

if not config.parallelism_group:
    config.parallelism_group = "shadow-memory"

if config.host_os == "NetBSD":
    config.substitutions.insert(0, ("%run", config.netbsd_noaslr_prefix))
