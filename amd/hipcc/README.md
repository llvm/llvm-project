# HIP compiler driver (hipcc)

## Table of Contents

<!-- toc -->

- [hipcc](#hipcc)
     * [Environment Variables](#envVar)
     * [Usage](#hipcc-usage)
     * [Building](#building)
     * [Testing](#testing)
     * [Linux](#linux)
     * [Windows](#windows)

<!-- tocstop -->

## <a name="hipcc"></a> hipcc

`hipcc` is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure.

It will pass-through options to the target compiler. The tools calling hipcc must ensure the compiler options are appropriate for the target compiler.

### <a name="envVar"></a> Environment Variables

The environment variable HIP_PLATFORM may be used to specify amd/nvidia:
- HIP_PLATFORM='amd' or HIP_PLATFORM='nvidia'.
- If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.

Other environment variable controls:
- HIP_PATH        : Path to HIP directory, default is one dir level above location of hipcc.
- CUDA_PATH       : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.
- HSA_PATH        : Path to HSA dir (defaults to ../../hsa relative to abs_path of hipcc). Used on AMD platforms only.
- HIP_ROCCLR_HOME : Path to HIP/ROCclr directory. Used on AMD platforms only.
- HIP_CLANG_PATH  : Path to HIP-Clang (default to ../../llvm/bin relative to hipcc's abs_path). Used on AMD platforms only.

### <a name="usage"></a> hipcc: usage

- WIP

### <a name="building"></a> hipcc: building

- WIP

### <a name="testing"></a> hipcc: testing

- WIP
