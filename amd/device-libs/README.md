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
sources in the amd-stg-open branch of AMD-modified llvm-project repository.
Use the following commands:

    git clone https://github.com/RadeonOpenCompute/llvm-project.git -b amd-stg-open llvm_amd
    cd llvm_amd
    mkdir -p build
    cd build
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm \
        -DLLVM_ENABLE_PROJECTS="clang;lld"    \
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86"  \
        ../llvm
    make


To build the library bitcodes, clone the amd_stg_open branch of this repository

    git clone https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git -b amd-stg-open

and from its top level run the following commands:

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

To create packages for the library:
   make package

## USING BITCODE LIBRARIES

The ROCm language compilers and runtimes automatically link the
required bitcode files invoked during the process of creating a code
object. clang will search for these libraries by default when
targeting amdhsa, in the default ROCm install location. To specify a
specific set of libraries, the --rocm-path argument can point to the
root directory where the bitcode libraries are installed, which is the
recommended way to link the libraries.

    $LLVM_BUILD/bin/clang -x cl -Xclang -finclude-default-header \
      -target amdgcn-amd-amdhsa -mcpu=gfx900 \
      --rocm-path=/srv/git/ROCm-Device-Libs/build/dist

These can be manually linked, but is generally not recommended. The
set of libraries linked should be in sync with the corresponding
compiler flags and target options. The default library linking can be
disabled with -nogpulib, and a manual linking invocation might look
like as follows:

    $LLVM_BUILD/bin/clang -x cl -Xclang -finclude-default-header \
        -nogpulib -target amdgcn-amd-amdhsa -mcpu=gfx900 \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/opencl/opencl.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/ocml/ocml.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/ockl/ockl.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/oclc/oclc_correctly_rounded_sqrt_off.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/oclc/oclc_daz_opt_off.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/oclc/oclc_finite_only_off.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/oclc/oclc_unsafe_math_off.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/oclc/oclc_wavefrontsize64_on.bc \
        -Xclang -mlink-bitcode-file -Xclang /srv/git/ROCm-Device-Libs/build/dist/amdgcn/bitcode/oclc/oclc_isa_version_900.bc \
        test.cl -o test.so

### USING FROM CMAKE

The bitcode libraries are exported as CMake targets, organized in a CMake
package. You can depend on this package using
`find_package(AMDDeviceLibs REQUIRED CONFIG)` after ensuring the
`CMAKE_PREFIX_PATH` includes either the build directory or install prefix of
the bitcode libraries. The package defines a variable
`AMD_DEVICE_LIBS_TARGETS` containing a list of the exported CMake
targets.

