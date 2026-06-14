# Device-profile drain tests: require an AMD GPU (and, implicitly, the amdgcn
# device profile runtime in the resource directory and a ROCm/HIP install).
if "amdgpu" not in config.available_features:
    config.unsupported = True
