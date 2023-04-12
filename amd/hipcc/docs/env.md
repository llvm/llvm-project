# Environment Variables

The environment variable HIP_PLATFORM may be used to specify amd/nvidia:
- HIP_PLATFORM='amd' or HIP_PLATFORM='nvidia'.
- If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.

Other environment variable controls:
- HIP_PATH        : Path to HIP directory, default is one dir level above location of hipcc.
- CUDA_PATH       : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.
- HSA_PATH        : Path to HSA dir (defaults to ../../hsa relative to abs_path of hipcc). Used on AMD platforms only.
- HIP_ROCCLR_HOME : Path to HIP/ROCclr directory. Used on AMD platforms only.
- HIP_CLANG_PATH  : Path to HIP-Clang (default to ../../llvm/bin relative to hipcc's abs_path). Used on AMD platforms only.
