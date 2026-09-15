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
config.name = "UBSan-" + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

default_ubsan_opts = list(config.default_sanitizer_opts)
# Choose between standalone and UBSan+ASan modes.
ubsan_lit_test_mode = get_required_attr(config, "ubsan_lit_test_mode")
if ubsan_lit_test_mode == "Standalone":
    config.available_features.add("ubsan-standalone")
    clang_ubsan_cflags = []
elif ubsan_lit_test_mode == "StandaloneStatic":
    config.available_features.add("ubsan-standalone-static")
    clang_ubsan_cflags = ["-static-libsan"]
elif ubsan_lit_test_mode == "AddressSanitizer":
    config.available_features.add("ubsan-asan")
    clang_ubsan_cflags = ["-fsanitize=address"]
    default_ubsan_opts += ["detect_leaks=0"]
elif ubsan_lit_test_mode == "MemorySanitizer":
    config.available_features.add("ubsan-msan")
    clang_ubsan_cflags = ["-fsanitize=memory"]
elif ubsan_lit_test_mode == "ThreadSanitizer":
    config.available_features.add("ubsan-tsan")
    clang_ubsan_cflags = ["-fsanitize=thread"]
elif ubsan_lit_test_mode == "TypeSanitizer":
    config.available_features.add("ubsan-tysan")
    clang_ubsan_cflags = ["-fsanitize=type"]
else:
    lit_config.fatal("Unknown UBSan test mode: %r" % ubsan_lit_test_mode)

# Platform-specific default for lit tests.
if config.target_arch == "s390x":
    # On SystemZ we need -mbackchain to make the fast unwinder work.
    clang_ubsan_cflags.append("-mbackchain")

default_ubsan_opts_str = ":".join(default_ubsan_opts)
if default_ubsan_opts_str:
    config.environment["UBSAN_OPTIONS"] = default_ubsan_opts_str
    default_ubsan_opts_str += ":"
# Substitution to setup UBSAN_OPTIONS in portable way.
config.substitutions.append(
    ("%env_ubsan_opts=", "env UBSAN_OPTIONS=" + default_ubsan_opts_str)
)


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


target_cflags = [get_required_attr(config, "target_cflags")]
clang_ubsan_cflags += target_cflags
clang_ubsan_cxxflags = config.cxx_mode_flags + clang_ubsan_cflags

# Define %clang and %clangxx substitutions to use in test RUN lines.
config.substitutions.append(("%clang ", build_invocation(clang_ubsan_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(clang_ubsan_cxxflags)))
config.substitutions.append(("%gmlt ", " ".join(config.debug_info_flags) + " "))

# Default test suffixes.
config.suffixes = [".c", ".cpp", ".m"]

# Check that the host supports UndefinedBehaviorSanitizer tests
if config.target_os not in [
    "Linux",
    "Darwin",
    "FreeBSD",
    "Windows",
    "NetBSD",
    "SunOS",
    "OpenBSD",
]:
    config.unsupported = True

config.excludes = ["Inputs"]

if ubsan_lit_test_mode in ["AddressSanitizer", "MemorySanitizer", "ThreadSanitizer"]:
    if not config.parallelism_group:
        config.parallelism_group = "shadow-memory"
    if config.target_os == "NetBSD":
        config.substitutions.insert(0, ("%run", config.netbsd_noaslr_prefix))

if ubsan_lit_test_mode == "Standalone":
    _amdgpu_ubsan_rt = any(
        os.path.isfile(
            os.path.join(
                config.compiler_rt_output_dir,
                "lib",
                triple,
                "libclang_rt.ubsan_standalone.a",
            )
        )
        for triple in ("amdgpu-amd-amdhsa", "amdgcn-amd-amdhsa")
    )
    if config.ubsan_can_run_hip and _amdgpu_ubsan_rt:
        config.available_features.add("ubsan-hip")
        hip_common = [
            "-xhip",
            "--offload-arch=" + config.ubsan_gpu_arch,
            "-nogpuinc",
            "-nogpulib",
            "-g",
            "-isystem",
            os.path.join(config.test_source_root, "Inputs"),
            "-include",
            "hip.h",
            "-fsanitize=undefined",
        ]
        hip_libs = [
            "-L" + config.ubsan_hip_lib_dir,
            "-lamdhip64",
            "-Wl,-rpath," + config.ubsan_hip_lib_dir,
        ]
        config.substitutions.append(
            ("%clang_ubsan_hip ", build_invocation(hip_common) + " ")
        )
        config.substitutions.append(("%hip_libs", " ".join(hip_libs)))
    if config.ubsan_can_run_openmp_offload and _amdgpu_ubsan_rt:
        config.available_features.add("ubsan-openmp-offload")
        omp_common = [
            "-fopenmp",
            "--offload-arch=" + config.ubsan_gpu_arch,
            "-g",
            "-frtlib-add-rpath",
            "-fsanitize=undefined",
        ]
        config.substitutions.append(
            ("%clang_ubsan_omp ", build_invocation(omp_common) + " ")
        )
    if config.ubsan_can_run_hip or config.ubsan_can_run_openmp_offload:
        lit_config.parallelism_groups["gpu"] = 1
