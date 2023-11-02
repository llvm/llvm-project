## OVERVIEW

ROCm Device libraries.

This subdirectory contains the sources and CMake build system for a
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

The build requires clang and several llvm development tools. This can
be built using the amd-stg-open branch of the RadeonOpenCompute
modified llvm-project repository where this subdirectory now lives,
but the upstream llvm-project should also work.

There are two different methods to build the device libraries; as a
standalone project or as an llvm external subproject.

For a standalone build, this will find preexisting clang and llvm
tools using the standard cmake search mechanisms. If you wish to use a
specific build, you can specify this with the CMAKE_PREFIX_PATH
variable:

    git clone https://github.com/RadeonOpenCompute/llvm-project.git -b amd-stg-open
    cd llvm-project/amd/device-libs

Then run the following commands:

    mkdir -p build
    cd build
    export LLVM_BUILD=... (path to LLVM build directory created previously)
    cmake -DCMAKE_PREFIX_PATH=$LLVM_BUILD ..
    make

To build as an llvm external project:

    LLVM_PROJECT_ROOT=llvm-project-rocm
    git clone https://github.com/RadeonOpenCompute/llvm-project.git -b amd-stg-open ${LLVM_PROJECT_ROOT}
    cd ${LLVM_PROJECT_ROOT}
    mkdir -p build
    cd build

    cmake ${LLVM_PROJECT_ROOT}/llvm -DCMAKE_BUILD_TYPE=Release \
          -DLLVM_ENABLE_PROJECTS="clang;lld" \
          -DLLVM_EXTERNAL_PROJECTS="device-libs" \
          -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR=/path/to/ROCm-Device-Libs

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

