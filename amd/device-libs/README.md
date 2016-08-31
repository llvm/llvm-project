## OVERVIEW

ROCm Device libraries are currently in early development and considered experimental and incomplete.

This repository contains the following libraries:

| **Name** | **Comments** | **Dependencies** |
| --- | --- | --- |
| irif | Interface to LLVM IR | |
| ocml | Open Compute Math library ([documentation](doc/OCML.md)) | irif |
| ockl | Open Compute Kernel library. | irif |
| opencl | OpenCL built-in library | ocml, ockl |
| hc | Heterogeneous Compute built-in library | ocml, ockl |

All libraries are compiled to LLVM Bitcode which can be linked. Note that libraries use specific AMDGPU intrinsics.

Refer to [LICENSE.TXT](LICENSE.TXT) for license information.

## BUILDING

To build it, use RadeonOpenCompute LLVM/LLD/Clang. Default branch on these
repositories is "amd-common", which contains AMD-specific codes yet upstreamed.

    git clone git@github.com:RadeonOpenCompute/llvm.git llvm_amd-common
    cd llvm_amd-common/tools
    git clone git@github.com:RadeonOpenCompute/lld.git lld
    git clone git@github.com:RadeonOpenCompute/clang.git clang
    cd ..
    mkdir -p build
    cd build
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm \
        -DLLVM_TARGET_TO_BUILD="AMDGPU;X86" \
        ..

Testing also requires amdhsacod utility from ROCm Runtime.

Use out-of-source CMake build and create separate directory to run CMake.

The following build steps are performed:

    mkdir -p build
    cd build
    export LLVM_BUILD=... (path to LLVM build)
    CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD -DAMDHSACOD=$HSA_DIR/bin/x86_64/amdhsacod ..
    make
    make install
    make test

## TESTING

Currently all tests are offline:
 * OpenCL source is compiled to LLVM bitcode
 * Test bitcode is linked to library bitcode with llvm-link
 * Clang OpenCL compiler is run on resulting bitcode, producing code object.
 * Resulting code object is passed to llvm-objdump and amdhsacod -test.

The output of tests (which includes AMDGPU disassembly) can be displayed by running ctest -VV in build directory.

Tests for OpenCL conformance kernels can be enabled by specifying -DOCL_CONFORMANCE_HOME=<path> to CMake, for example,
  cmake ... -DOCL_CONFORMANCE_HOME=/srv/hsa/drivers/opencl/tests/extra/hsa/ocl/conformance/1.2
