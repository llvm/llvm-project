import glob
import os


def getRoot(config):
    if not config.parent:
        return config
    return getRoot(config.parent)


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


root = getRoot(config)

# AMDGPU ASan tests are only run with the dynamic ASan runtime (-shared-libasan).
if "asan-static-runtime" in root.available_features:
    config.unsupported = True
elif root.target_os != "Linux":
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
            rocm_lib = rocm_lib_dir(rocm_root)
            rocm_include = os.path.join(rocm_root, "include")
            config.substitutions.append(("%rocm_root", rocm_root))
            config.substitutions.append(("%rocm_include", rocm_include))
            config.substitutions.append(("%rocm_lib", rocm_lib))
            config.substitutions.append(("%compiler_rt_libdir", rt_libdir))
