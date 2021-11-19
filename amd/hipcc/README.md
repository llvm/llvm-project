# HIP compiler driver (HIPCC)

HIPCC is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure.

It will pass-through options to the target compiler. The tools calling HIPCC must ensure the compiler options are appropriate for the target compiler.

The environment variable HIP_PLATFORM may be used to specify amd/nvidia:
- HIP_PLATFORM='amd' or HIP_PLATFORM='nvidia'.
- If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.

Other environment variable controls:
- HIP_PATH        : Path to HIP directory, default is one dir level above location of HIPCC.
- CUDA_PATH       : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.
- HSA_PATH        : Path to HSA dir (defaults to ../../hsa relative to abs_path of HIPCC). Used on AMD platforms only.
- HIP_ROCCLR_HOME : Path to HIP/ROCclr directory. Used on AMD platforms only.
- HIP_CLANG_PATH  : Path to HIP-Clang (default to ../../llvm/bin relative to HIPCC's abs_path). Used on AMD platforms only.