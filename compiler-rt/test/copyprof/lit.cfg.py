import os

# Sets a proper name for the test suite that includes the target triple.
config.name = "CopyProf" + config.name_suffix
# Lets llvm-lit find all tests beneath the current directory.
config.test_source_root = os.path.dirname(__file__)

# Create a substitution so lit tests can use `%env_copyprof_opts=` to parameterize CopyProf via
# the `COPYPROF_OPTIONS` environment variable.
default_copyprof_opts = list(config.default_sanitizer_opts)
default_copyprof_opts_str = ":".join(default_copyprof_opts)
if default_copyprof_opts_str:
    config.environment["COPYPROF_OPTIONS"] = default_copyprof_opts_str
    default_copyprof_opts_str += ":"
config.substitutions.append(
    ("%env_copyprof_opts=", "env COPYPROF_OPTIONS=" + default_copyprof_opts_str)
)

# Setup default compiler flags used with -fcopyprof option.
clang_copyprof_cflags = ["-fcopyprof"] + [config.target_cflags]
clang_copyprof_cxxflags = config.cxx_mode_flags + clang_copyprof_cflags


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


config.substitutions.append(("%clang_copyprof ", build_invocation(clang_copyprof_cflags)))
config.substitutions.append(("%clangxx_copyprof ", build_invocation(clang_copyprof_cxxflags)))

# Default test suffixes.
config.suffixes = [".c", ".cpp"]

# CopyProf tests are currently supported on Linux only.
if not (config.target_os in ["Linux"] and config.target_arch in ["x86_64"]):
    config.unsupported = True
