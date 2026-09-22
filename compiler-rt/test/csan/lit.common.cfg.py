# -*- Python -*-

import os


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal("No attribute %r in test configuration!" % attr_name)
    return attr_value


config.name = "CSan-" + config.name_suffix
config.test_source_root = os.path.dirname(__file__)
config.suffixes = [".c", ".cpp"]


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


target_cflags = [get_required_attr(config, "target_cflags")]
config.substitutions.append(("%clang ", build_invocation(target_cflags)))
config.substitutions.append(
    ("%clangxx ", build_invocation(config.cxx_mode_flags + target_cflags))
)
config.substitutions.append(
    ("%clang_csan ", build_invocation(target_cflags + ["-fsanitize=concurrency"]))
)
config.substitutions.append(
    (
        "%clangxx_csan ",
        build_invocation(
            config.cxx_mode_flags + target_cflags + ["-fsanitize=concurrency"]
        ),
    )
)

if config.target_os not in ["Linux"]:
    config.unsupported = True

if "csan" in config.gpu_runtimes:
    if "hip" in config.available_features:
        config.available_features.add("csan-hip")
    if "openmp-offload" in config.available_features:
        config.available_features.add("csan-openmp-offload")


def add_csan_substitution(name, base):
    for pattern, replacement in config.substitutions:
        if pattern == base:
            config.substitutions.append(
                (name, replacement.rstrip() + " -fsanitize=concurrency ")
            )
            return
    lit_config.fatal("Missing substitution %r" % base)


if "csan-hip" in config.available_features:
    add_csan_substitution("%clang_hip_csan ", "%clang_hip ")
if "csan-openmp-offload" in config.available_features:
    add_csan_substitution("%clang_omp_offload_csan ", "%clang_omp_offload ")
