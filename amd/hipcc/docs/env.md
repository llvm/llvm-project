# Environment Variables

The environment variable HIP_PLATFORM may be used to specify amd/nvidia:
- HIP_PLATFORM='amd' or HIP_PLATFORM='nvidia'.
- If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.

Other environment variable controls:
- CUDA_PATH       : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.
