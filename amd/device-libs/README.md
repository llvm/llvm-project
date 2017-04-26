## OVERVIEW

ROCm Device libraries.

This repository contains the following libraries:

| **Name** | **Comments** | **Dependencies** |
| --- | --- | --- |
| irif | Interface to LLVM IR | |
| ocml | Open Compute Math library ([documentation](doc/OCML.md)) | irif |
| oclc | Open Compute library controls ([documentation](doc/OCML.md#controls)) | |
| ockl | Open Compute Kernel library. | irif |
| opencl | OpenCL built-in library | ocml, ockl |
| hc | Heterogeneous Compute built-in library | ocml, ockl |

All libraries are compiled to LLVM Bitcode which can be linked. Note that libraries use specific AMDGPU intrinsics.

Refer to [LICENSE.TXT](LICENSE.TXT) for license information.

## BUILDING

To build it, use RadeonOpenCompute LLVM/LLD/Clang. Default branch on these
repositories is "amd-common", which may contain AMD-specific codes yet
upstreamed.

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
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
        ..

Testing also requires amdhsacod utility from ROCm Runtime.

Use out-of-source CMake build and create separate directory to run CMake.

The following build steps are performed:

    mkdir -p build
    cd build
    export LLVM_BUILD=... (path to LLVM build)
    CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD -DAMDHSACOD=$HSA_DIR/bin/x86_64/amdhsacod ..
    make

It is also possible to use compiler that only has AMDGPU target enabled if you build prepare-builtins separately
with host compiler and pass explicit target option to CMake:

    export LLVM_BUILD=... (path to LLVM build)
    # Build prepare-builtins
    cd utils
    mkdir build
    cd build
    cmake -DLLVM_DIR=$LLVM_BUILD ..
    make
    # Build bitcode libraries
    cd ../..
    mkdir build
    cd build
    CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD -DAMDHSACOD=$HSA_DIR/bin/x86_64/amdhsacod -DCMAKE_C_FLAGS="-target amdgcn--amdhsa" -DCMAKE_CXX_FLAGS="-target amdgcn--amdhsa" -DPREPARE_BUILTINS=`cd ../utils/build/prepare-builtins/; pwd`/prepare-builtins ..

To install artifacts:
    make install

To run offline tests:
    make test

To create packages for the library:
   make package

## USING BITCODE LIBRARIES

The bitcode libraries should be linked to user bitcode (obtained from source) *before* final code generation
with llvm-link or -mlink-bitcode-file option of clang.

For OpenCL, the list of bitcode libraries includes opencl, its dependencies (ocml, ockl, irif)
and oclc control libraries selected according to OpenCL compilation mode.  Assuming that the build
of this repository was done in /srv/git/ROCm-Device-Libs/build, the following command line
shows how to compile simple OpenCL source test.cl into code object test.so:

    clang -x cl -Xclang -finclude-default-header \
        -target amdgcn--amdhsa -mcpu=fiji \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/opencl/opencl.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ocml/ocml.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ockl/ockl.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_correctly_rounded_sqrt_off.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_daz_opt_off.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_finite_only_off.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_isa_version_803.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_unsafe_math_off.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/irif/irif.amdgcn.bc \
        test.cl -o test.so

## TESTING

Currently all tests are offline:
 * OpenCL source is compiled to LLVM bitcode
 * Test bitcode is linked to library bitcode with llvm-link
 * Clang OpenCL compiler is run on resulting bitcode, producing code object.
 * Resulting code object is passed to llvm-objdump and amdhsacod -test.

The output of tests (which includes AMDGPU disassembly) can be displayed by running ctest -VV in build directory.

Tests for OpenCL conformance kernels can be enabled by specifying -DOCL_CONFORMANCE_HOME=<path> to CMake, for example,
  cmake ... -DOCL_CONFORMANCE_HOME=/srv/hsa/drivers/opencl/tests/extra/hsa/ocl/conformance/1.2
