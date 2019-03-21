## OVERVIEW

ROCm Device libraries.

This repository contains the sources and CMake build system for a
set of AMD specific device-side language runtime libraries.  Specifically:

| **Name** | **Comments** | **Dependencies** |
| --- | --- | --- |
| oclc* | Open Compute library controls ([documentation](doc/OCML.md#controls)) | |
| ocml | Open Compute Math library ([documentation](doc/OCML.md)) | oclc* |
| ockl | Open Compute Kernel library ([documentation](doc/OCKL.md)) | oclc* |
| opencl | OpenCL built-in library | ocml, ockl, oclc* |
| hip | HIP built-in library | ocml, ockl, oclc* |
| hc | Heterogeneous Compute built-in library | ocml, ockl, oclc* |

Refer to [LICENSE.TXT](LICENSE.TXT) for license information.

## BUILDING

The library sources should be compiled using a clang compiler built from
sources in the amd-common branch of AMD modified clang, llvm, and lld repositories
using the following commands:

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


To build the library bitcodes, from the top level of this repository
run the following commands:

    mkdir -p build
    cd build
    export LLVM_BUILD=... (path to LLVM build directory created above)
    CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD ..
    make

It is also possible to use a compiler that only has AMDGPU target enabled if you build prepare-builtins separately
with the regular host compiler and pass explicit target option to CMake:

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
    CC=$LLVM_BUILD/bin/clang cmake -DLLVM_DIR=$LLVM_BUILD -DPREPARE_BUILTINS=`cd ../utils/build/prepare-builtins/; pwd`/prepare-builtins ..

Testing requires the amdhsacod utility from ROCm Runtime.

To install artifacts:
    make install

To run offline tests:
    make test

To create packages for the library:
   make package

## USING BITCODE LIBRARIES
 
The ROCm language runtimes automatically add the required bitcode files during the
LLVM linking stage invoked during the process of creating a code object.  There are options
to display the exact commands executed, but an approximation of the command the OpenCL
runtime might use is as follows:

    $LLVM_BUILD/bin/clang -x cl -Xclang -finclude-default-header \
        -target amdgcn-amd-amdhsa -mcpu=gfx803 \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/opencl/opencl.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ocml/ocml.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/ockl/ockl.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_correctly_rounded_sqrt_off.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_daz_opt_off.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_finite_only_off.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_isa_version_803.amdgcn.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/oclc/oclc_unsafe_math_off.amdgcn.bc \
        test.cl -o test.so

### USING FROM CMAKE

The bitcode libraries are exported as CMake targets, organized in a CMake
package. You can depend on this package using
`find_package(AMDDeviceLibs REQUIRED CONFIG)` after ensuring the
`CMAKE_PREFIX_PATH` includes either the build directory or install prefix of
the bitcode libraries. The package defines a variable
`AMD_DEVICE_LIBS_TARGETS` containing a list of the exported CMake
targets.

## TESTING

Currently all tests are offline:
 * OpenCL source is compiled to LLVM bitcode
 * Test bitcode is linked to library bitcode with llvm-link
 * Clang OpenCL compiler is run on resulting bitcode, producing code object.
 * Resulting code object is passed to llvm-objdump and amdhsacod -test.

The output of tests (which includes AMDGPU disassembly) can be displayed by running ctest -VV in build directory.

Tests for OpenCL conformance kernels can be enabled by specifying -DOCL_CONFORMANCE_HOME=<path> to CMake, for example,
  cmake ... -DOCL_CONFORMANCE_HOME=/srv/hsa/drivers/opencl/tests/extra/hsa/ocl/conformance/1.2
