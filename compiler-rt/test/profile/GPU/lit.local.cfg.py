# HIP device-PGO tests for host-shadowed (statically linked) kernels: require an
# AMD GPU plus a ROCm/HIP install (features set in ../lit.cfg.py).
if not {"hip", "amdgpu"}.issubset(config.available_features):
    config.unsupported = True
else:
    # Tests share the GPU(s) and pin HIP_VISIBLE_DEVICES; serialize them.
    config.parallelism_group = "gpu"
