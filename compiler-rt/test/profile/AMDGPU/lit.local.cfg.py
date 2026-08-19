# Device-profile drain tests: require an AMD GPU (and, implicitly, the amdgcn
# device profile runtime in the resource directory and a ROCm/HIP install).
if not {"hip", "amdgpu"}.issubset(config.available_features):
    config.unsupported = True
else:
    # Tests share the GPU(s) and pin HIP_VISIBLE_DEVICES; serialize them.
    config.parallelism_group = "gpu"
