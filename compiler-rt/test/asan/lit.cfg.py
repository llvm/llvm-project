# -*- Python -*-

import os
import platform
import re
import shlex

import lit.formats


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value == None:
        lit_config.fatal(
            "No attribute %r in test configuration! You may need to run "
            "tests from your build directory or add this attribute "
            "to lit.site.cfg.py " % attr_name
        )
    return attr_value

# Setup config name.
config.name = "AddressSanitizer" + config.name_suffix

# Platform-specific default ASAN_OPTIONS for lit tests.
default_asan_opts = list(config.default_sanitizer_opts)

# On Darwin, leak checking is not enabled by default. Enable on macOS
# tests to prevent regressions
if config.host_os == "Darwin" and config.apple_platform == "osx":
    default_asan_opts += ["detect_leaks=1"]

default_asan_opts_str = ":".join(default_asan_opts)
if default_asan_opts_str:
    config.environment["ASAN_OPTIONS"] = default_asan_opts_str
    default_asan_opts_str += ":"
config.substitutions.append(
    ("%env_asan_opts=", "env ASAN_OPTIONS=" + default_asan_opts_str)
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
clang_asan_static_cflags = (
    [
        "-fsanitize=address",
        "-mno-omit-leaf-frame-pointer",
        "-fno-omit-frame-pointer",
        "-fno-optimize-sibling-calls",
    ]
    + config.debug_info_flags
    + target_cflags
)
if config.target_arch == "s390x":
    clang_asan_static_cflags.append("-mbackchain")
clang_asan_static_cxxflags = config.cxx_mode_flags + clang_asan_static_cflags

target_is_msvc = bool(re.match(r".*-windows-msvc$", config.target_triple))

asan_dynamic_flags = []
if config.asan_dynamic:
    asan_dynamic_flags = ["-shared-libasan"]
    if platform.system() == "Windows" and target_is_msvc:
        # On MSVC target, we need to simulate "clang-cl /MD" on the clang driver side.
        asan_dynamic_flags += [
            "-D_MT",
            "-D_DLL",
            "-Wl,-nodefaultlib:libcmt,-defaultlib:msvcrt,-defaultlib:oldnames",
        ]
    elif platform.system() == "FreeBSD":
        # On FreeBSD, we need to add -pthread to ensure pthread functions are available.
        asan_dynamic_flags += ["-pthread"]
    config.available_features.add("asan-dynamic-runtime")
else:
    config.available_features.add("asan-static-runtime")
clang_asan_cflags = clang_asan_static_cflags + asan_dynamic_flags
clang_asan_cxxflags = clang_asan_static_cxxflags + asan_dynamic_flags

# Add win32-(static|dynamic)-asan features to mark tests as passing or failing
# in those modes. lit doesn't support logical feature test combinations.
if platform.system() == "Windows":
    if config.asan_dynamic:
        win_runtime_feature = "win32-dynamic-asan"
    else:
        win_runtime_feature = "win32-static-asan"
    config.available_features.add(win_runtime_feature)


def build_invocation(compile_flags, with_lto=False):
    lto_flags = []
    if with_lto and config.lto_supported:
        lto_flags += config.lto_flags

    return " " + " ".join([config.clang] + lto_flags + compile_flags) + " "


config.substitutions.append(("%clang ", build_invocation(target_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(target_cxxflags)))
config.substitutions.append(("%clang_asan ", build_invocation(clang_asan_cflags)))
config.substitutions.append(("%clangxx_asan ", build_invocation(clang_asan_cxxflags)))
config.substitutions.append(
    ("%clang_asan_lto ", build_invocation(clang_asan_cflags, True))
)
config.substitutions.append(
    ("%clangxx_asan_lto ", build_invocation(clang_asan_cxxflags, True))
)
if config.asan_dynamic:
    if config.host_os in ["Linux", "FreeBSD", "NetBSD", "SunOS"]:
        shared_libasan_path = os.path.join(
            config.compiler_rt_libdir,
            "libclang_rt.asan{}.so".format(config.target_suffix),
        )
    elif config.host_os == "Darwin":
        shared_libasan_path = os.path.join(
            config.compiler_rt_libdir,
            "libclang_rt.asan_{}_dynamic.dylib".format(config.apple_platform),
        )
    else:
        lit_config.warning(
            "%shared_libasan substitution not set but dynamic ASan is available."
        )
        shared_libasan_path = None

    if shared_libasan_path is not None:
        config.substitutions.append(("%shared_libasan", shared_libasan_path))
    config.substitutions.append(
        ("%clang_asan_static ", build_invocation(clang_asan_static_cflags))
    )
    config.substitutions.append(
        ("%clangxx_asan_static ", build_invocation(clang_asan_static_cxxflags))
    )

if platform.system() == "Windows":
    # MSVC-specific tests might also use the clang-cl.exe driver.
    if target_is_msvc:
        clang_cl_cxxflags = [
            "-Wno-deprecated-declarations",
            "-WX",
            "-D_HAS_EXCEPTIONS=0",
            "-Zi",
        ] + target_cflags
        clang_cl_asan_cxxflags = ["-fsanitize=address"] + clang_cl_cxxflags
        if config.asan_dynamic:
            clang_cl_asan_cxxflags.append("-MD")

        clang_cl_invocation = build_invocation(clang_cl_cxxflags)
        clang_cl_invocation = clang_cl_invocation.replace("clang.exe", "clang-cl.exe")
        config.substitutions.append(("%clang_cl ", clang_cl_invocation))

        clang_cl_asan_invocation = build_invocation(clang_cl_asan_cxxflags)
        clang_cl_asan_invocation = clang_cl_asan_invocation.replace(
            "clang.exe", "clang-cl.exe"
        )
        config.substitutions.append(("%clang_cl_asan ", clang_cl_asan_invocation))
        config.substitutions.append(("%clang_cl_nocxx_asan ", clang_cl_asan_invocation))
        config.substitutions.append(("%Od", "-Od"))
        config.substitutions.append(("%Fe", "-Fe"))
        config.substitutions.append(("%LD", "-LD"))
        config.substitutions.append(("%MD", "-MD"))
        config.substitutions.append(("%MT", "-MT"))
        config.substitutions.append(("%Gw", "-Gw"))

        base_lib = os.path.join(
            config.compiler_rt_libdir, "clang_rt.asan%%s%s.lib" % config.target_suffix
        )
        config.substitutions.append(("%asan_lib", base_lib % ""))
        config.substitutions.append(("%asan_cxx_lib", base_lib % "_cxx"))
        config.substitutions.append(("%asan_dll_thunk", base_lib % "_dll_thunk"))
    else:
        # To make some of these tests work on MinGW target without changing their
        # behaviour for MSVC target, substitute clang-cl flags with gcc-like ones.
        config.substitutions.append(("%clang_cl ", build_invocation(target_cxxflags)))
        config.substitutions.append(
            ("%clang_cl_asan ", build_invocation(clang_asan_cxxflags))
        )
        config.substitutions.append(
            ("%clang_cl_nocxx_asan ", build_invocation(clang_asan_cflags))
        )
        config.substitutions.append(("%Od", "-O0"))
        config.substitutions.append(("%Fe", "-o"))
        config.substitutions.append(("%LD", "-shared"))
        config.substitutions.append(("%MD", ""))
        config.substitutions.append(("%MT", ""))
        config.substitutions.append(("%Gw", "-fdata-sections"))

# FIXME: De-hardcode this path.
asan_source_dir = os.path.join(
    get_required_attr(config, "compiler_rt_src_root"), "lib", "asan"
)
python_exec = shlex.quote(get_required_attr(config, "python_executable"))
# Setup path to asan_symbolize.py script.
asan_symbolize = os.path.join(asan_source_dir, "scripts", "asan_symbolize.py")
if not os.path.exists(asan_symbolize):
    lit_config.fatal("Can't find script on path %r" % asan_symbolize)
config.substitutions.append(
    ("%asan_symbolize", python_exec + " " + asan_symbolize + " ")
)
# Setup path to sancov.py script.
sanitizer_common_source_dir = os.path.join(
    get_required_attr(config, "compiler_rt_src_root"), "lib", "sanitizer_common"
)
sancov = os.path.join(sanitizer_common_source_dir, "scripts", "sancov.py")
if not os.path.exists(sancov):
    lit_config.fatal("Can't find script on path %r" % sancov)
config.substitutions.append(("%sancov ", python_exec + " " + sancov + " "))

# Determine kernel bitness
if config.host_arch.find("64") != -1 and not config.android:
    kernel_bits = "64"
else:
    kernel_bits = "32"

config.substitutions.append(
    ("CHECK-%kernel_bits", ("CHECK-kernel-" + kernel_bits + "-bits"))
)

config.substitutions.append(("%libdl", libdl_flag))

config.available_features.add("asan-" + config.bits + "-bits")

# Fast unwinder doesn't work with Thumb
if not config.arm_thumb:
    config.available_features.add("fast-unwinder-works")

# Turn on leak detection on 64-bit Linux.
leak_detection_android = (
    config.android
    and "android-thread-properties-api" in config.available_features
    and (config.target_arch in ["x86_64", "i386", "i686", "aarch64"])
)
leak_detection_linux = (
    (config.host_os == "Linux")
    and (not config.android)
    and (config.target_arch in ["x86_64", "i386", "riscv64", "loongarch64"])
)
leak_detection_mac = (config.host_os == "Darwin") and (config.apple_platform == "osx")
leak_detection_netbsd = (config.host_os == "NetBSD") and (
    config.target_arch in ["x86_64", "i386"]
)
if (
    leak_detection_android
    or leak_detection_linux
    or leak_detection_mac
    or leak_detection_netbsd
):
    config.available_features.add("leak-detection")

# Add the RT libdir to PATH directly so that we can successfully run the gtest
# binary to list its tests.
if config.host_os == "Windows" and config.asan_dynamic:
    os.environ["PATH"] = os.path.pathsep.join(
        [config.compiler_rt_libdir, os.environ.get("PATH", "")]
    )

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
if config.host_os not in ["Linux", "Darwin", "FreeBSD", "SunOS", "Windows", "NetBSD"]:
    config.unsupported = True

if not config.parallelism_group:
    config.parallelism_group = "shadow-memory"

if config.host_os == "NetBSD":
    config.substitutions.insert(0, ("%run", config.netbsd_noaslr_prefix))

# Find ROCM runtime and compiler paths only
# when built with -DSANITIZER_AMDGPU=1
def configure_rocm(config, test_rocm_path):
    if (not os.path.isdir(test_rocm_path)):
        print("no directory found")
        test_rocm_path = os.path.join('/opt','rocm')
        if (not os.path.isdir(test_rocm_path)):
            test_rocm_path = os.path.abspath(os.path.join(config.llvm_install_dir, os.pardir))
            if (not os.path.isdir(test_rocm_path)):
                sys.exit("ROCM installation not found, try exporting ASAN_TEST_ROCM variable")

    test_device_libs  = os.path.join(test_rocm_path, 'amdgcn', 'bitcode')
    test_hip_path     = os.path.join(test_rocm_path, 'hip')
    hipcc             = os.path.join(test_hip_path, 'bin', 'hipcc')

    build_clang = getattr(config, 'clang', None)
    build_clang = build_clang.lstrip()
    build_clang = build_clang.rstrip()
    test_clang_path = os.path.dirname(build_clang)

    def hip_build_invocation(hipcc, compile_flags):
        return ' ' + ' '.join([hipcc] + compile_flags) + ' ' # append extra space to avoid concat issue in shell

    hipcxx_sanitize_options = ["-fsanitize=address", "-shared-libsan", "-fgpu-sanitize"]

    config.substitutions.append(
        ('%hipcompiler',
        hip_build_invocation(hipcc, config.cxx_mode_flags + [config.target_cflags] + hipcxx_sanitize_options)))

    #ROCM SPECIFIC ENVIRONMENT VARIABLES
    device_library_path    = 'DEVICE_LIB_PATH=' + test_device_libs
    hip_path               = 'HIP_PATH='        + test_hip_path
    rocm_path              = 'ROCM_PATH='       + test_rocm_path
    clang_path             = 'HIP_CLANG_PATH='  + test_clang_path
    rocm_environment       = [device_library_path, hip_path, rocm_path, clang_path]
    export_rocm_components = 'export ' + ' '.join(rocm_environment)
    config.substitutions.append(('%ROCM_ENV', export_rocm_components))
    config.suffixes.append('.hip')

test_rocm_path = os.environ.get('ASAN_TEST_ROCM','null')
if config.support_amd_offload_tests == 'true':
    configure_rocm(config, test_rocm_path)
