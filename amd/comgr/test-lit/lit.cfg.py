import glob
import os
import platform
import sys
import subprocess
import tempfile

import lit.formats

config.name = "Comgr"
config.suffixes = {".hip", ".cl", ".c", ".cpp"}
config.test_format = lit.formats.ShTest(
    execute_external=True, force_execute_external=True
)

config.excludes = ["comgr-sources", "Inputs"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.my_obj_root

if config.comgr_spirv_backend_available:
    config.available_features.add("comgr-has-spirv-backend")
if config.comgr_spirv_translator_available:
    config.available_features.add("comgr-has-spirv-translator")
if config.comgr_amdgpu_target_available:
    config.available_features.add("comgr-has-amdgpu-target")

# The AMDGPU device AddressSanitizer runtime (libclang_rt.asan.a for
# amdgcn-amd-amdhsa) is a separately built artifact. The asan tests link it,
# so guard them behind its presence in the clang resource dir; builds without
# it (e.g. reduced builds that omit compiler-rt) skip rather than fail.
if glob.glob(
    os.path.join(
        config.llvm_tools_dir,
        os.pardir,
        "lib",
        "clang",
        "*",
        "lib",
        "amdgcn-amd-amdhsa",
        "libclang_rt.asan.a",
    )
):
    config.available_features.add("comgr-has-amdgpu-asan-runtime")

try:
    with open(config.comgr_resource_dir_file, encoding="utf-8") as source:
        embedded_resources = source.read()
except OSError:
    embedded_resources = ""

if '"lib/amdgcn-amd-amdhsa/libclang_rt.profile.a"' in embedded_resources:
    config.available_features.add("comgr-has-amdgpu-profile-runtime")

if (
    '"lib/amdgcn-amd-amdhsa/libclang_rt.asan.a"' in embedded_resources
    and '"lib/amdgcn-amd-amdhsa/libclang_rt.asan_static.a"' in embedded_resources
):
    config.available_features.add("comgr-has-embedded-amdgpu-asan-runtime")


# spirv-to-reloc-debuginfo checks that comgr forwards
# -amdgpu-spill-cfi-saved-regs, which the AMD clang driver embeds for -g
# amdgcnspirv compiles. That is an AMD downstream driver diff (not upstream),
# so probe the driver and guard the test; builds whose clang lacks it skip.
def _clang_embeds_debuginfo_cfi():
    clang = os.path.join(config.llvm_tools_dir, "clang")
    if not os.path.exists(clang):
        return False
    src = None
    try:
        fd, src = tempfile.mkstemp(suffix=".hip")
        os.write(fd, b"__attribute__((global)) void k(float *p) { *p = 1.0f; }\n")
        os.close(fd)
        out = subprocess.run(
            [
                clang,
                "-x",
                "hip",
                "--offload-arch=amdgcnspirv",
                "-nogpulib",
                "-nogpuinc",
                "--offload-device-only",
                "-O3",
                "-g",
                "-c",
                "-###",
                src,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        return b"amdgpu-spill-cfi-saved-regs" in out.stdout
    except Exception:
        return False
    finally:
        if src and os.path.exists(src):
            os.unlink(src)


if _clang_embeds_debuginfo_cfi():
    config.available_features.add("comgr-clang-embeds-debuginfo-cfi")

if platform.system() == "Windows":
    config.available_features.add("system-windows")
elif platform.system() == "Linux":
    config.available_features.add("system-linux")
    if os.path.exists("/usr/include/c++/v1/cstddef") or os.path.exists(
        "/usr/local/include/c++/v1/cstddef"
    ):
        config.available_features.add("system-libcxx")

# By default, disable the cache for the tests.
# Test for the cache must explicitly enable this variable.
config.environment['AMD_COMGR_CACHE'] = "0"

# Resolve tool paths at configure time with forward slashes.  On Windows,
# os.path.join may return paths with backslashes, which break when written
# into bash scripts (e.g. "bin\clang" -> "binclang").
def _fwd(*parts):
    return os.path.join(*parts).replace("\\", "/")

# %-prefixed substitutions for LLVM tools (used as %clang, %llvm-dis, etc.)
config.substitutions.append(("%clang", _fwd(config.llvm_tools_dir, "clang")))
config.substitutions.append(("%llvm-as", _fwd(config.llvm_tools_dir, "llvm-as")))
config.substitutions.append(("%llvm-dis", _fwd(config.llvm_tools_dir, "llvm-dis")))
config.substitutions.append(
    ("%llvm-objcopy", _fwd(config.llvm_tools_dir, "llvm-objcopy"))
)
config.substitutions.append(
    ("%llvm-objdump", _fwd(config.llvm_tools_dir, "llvm-objdump"))
)
config.substitutions.append(
    ("%llvm-readelf", _fwd(config.llvm_tools_dir, "llvm-readelf"))
)
config.substitutions.append(
    ("%llvm-readobj", _fwd(config.llvm_tools_dir, "llvm-readobj"))
)
config.substitutions.append(("%ld.lld", _fwd(config.llvm_tools_dir, "ld.lld")))
config.substitutions.append(("%yaml2obj", _fwd(config.llvm_tools_dir, "yaml2obj")))
config.substitutions.append(("%FileCheck", _fwd(config.llvm_tools_dir, "FileCheck")))
config.substitutions.append(
    ("%amd-llvm-spirv", _fwd(config.llvm_tools_dir, "amd-llvm-spirv"))
)

# Interpreter used to run Python test helpers (e.g. enumerate-isa-check.py).
config.substitutions.append(("%python", _fwd(sys.executable)))
