# Link against ROCm's HSA runtime. Tests under TestCases/AMDGPU run only when
# lit finds a ROCm install (see lit.local.cfg.py): $ROCM_PATH or /opt/rocm,
# with include/hsa/hsa.h and libhsa-runtime64. Compiler-rt must be built with
# SANITIZER_AMDGPU enabled. The suite uses the dynamic ASan runtime only.

import glob
import os


def getRoot(config):
    if not config.parent:
        return config
    return getRoot(config.parent)


def walk_config_attr(cfg, name):
    """Return the first defined attribute `name` walking cfg -> parents."""
    while cfg is not None:
        if hasattr(cfg, name):
            val = getattr(cfg, name)
            if val is not None:
                return val
        cfg = cfg.parent
    return None


def rocm_lib_dir(rocm_root):
    """Return lib or lib64 under rocm_root that provides libhsa-runtime64."""
    for libname in ("lib", "lib64"):
        libdir = os.path.join(rocm_root, libname)
        if not os.path.isdir(libdir):
            continue
        if glob.glob(os.path.join(libdir, "libhsa-runtime64.so*")):
            return libdir
    return None


def rocm_is_available(rocm_root):
    if not rocm_root or not os.path.isdir(rocm_root):
        return False
    hsa_h = os.path.join(rocm_root, "include", "hsa", "hsa.h")
    if not os.path.isfile(hsa_h):
        return False
    return rocm_lib_dir(rocm_root) is not None


def append_detect_leaks_off_asan_subst(substitutions):
    """Append detect_leaks=0 to %env_asan_opts= / %export_asan_opts= default prefixes."""

    def patch_body(body):
        if body:
            return (
                body + "detect_leaks=0:"
                if body.endswith(":")
                else body + ":detect_leaks=0:"
            )
        return "detect_leaks=0:"

    pairs = (
        ("%env_asan_opts=", "env ASAN_OPTIONS="),
        ("%export_asan_opts=", "export ASAN_OPTIONS="),
    )
    for i, (pat, repl) in enumerate(substitutions):
        for sub_pat, opt_prefix in pairs:
            if pat == sub_pat and repl.startswith(opt_prefix):
                body = repl[len(opt_prefix) :]
                substitutions[i] = (pat, opt_prefix + patch_body(body))


root = getRoot(config)
# AMDGPU ASan tests are only run with the dynamic ASan runtime (-shared-libasan).
if "asan-static-runtime" in root.available_features:
    config.unsupported = True
elif root.target_os != "Linux":
    config.unsupported = True
elif walk_config_attr(config, "bits") == "32" or walk_config_attr(
    config, "target_arch"
) in ("i386", "i686"):
    # ROCm libhsa-runtime64.so is 64-bit only (link fails: incompatible with elf32-i386).
    config.unsupported = True
else:
    rocm_root = os.environ.get("ROCM_PATH", "/opt/rocm")
    if not rocm_is_available(rocm_root):
        config.unsupported = True
    else:
        # Dynamic ASan (-shared-libasan) adds libclang_rt.asan*.so as DT_NEEDED; embed
        # the host compiler-rt lib dir in RUNPATH so the loader finds it (same path as
        # LD_LIBRARY_PATH in lit.common.cfg.py, but explicit in the linked binary).
        rt_libdir = getattr(root, "compiler_rt_libdir", None)
        if not rt_libdir or not os.path.isdir(rt_libdir):
            config.unsupported = True
        elif not glob.glob(os.path.join(rt_libdir, "libclang_rt.asan*.so")):
            config.unsupported = True
        else:
            config.available_features.add("rocm")
            # Linux ASan defaults to leak detection; disable for ROCm/HSA tests.
            _asan = config.environment.get("ASAN_OPTIONS", "")
            if _asan:
                config.environment["ASAN_OPTIONS"] = _asan + ":detect_leaks=0"
            else:
                config.environment["ASAN_OPTIONS"] = "detect_leaks=0"
            append_detect_leaks_off_asan_subst(config.substitutions)
            rocm_lib = rocm_lib_dir(rocm_root)
            rocm_include = os.path.join(rocm_root, "include")
            config.substitutions.append(("%rocm_root", rocm_root))
            config.substitutions.append(("%rocm_include", rocm_include))
            config.substitutions.append(("%rocm_lib", rocm_lib))
            config.substitutions.append(("%compiler_rt_libdir", rt_libdir))
