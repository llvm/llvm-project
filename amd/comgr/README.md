Code Object Manager (Comgr)
===========================

The Comgr library provides APIs for compiling and inspecting AMDGPU code
objects. The API is documented in the [header file](include/amd_comgr.h).

Build the Code Object Manager
-----------------------------

Comgr depends on LLVM, and optionally depends on
[AMDDeviceLibs](https://github.com/RadeonOpenCompute/ROCm-Device-Libs). The
`CMAKE_PREFIX_PATH` must include either the build directory or install prefix
of both of these components. Comgr also depends on yaml-cpp, but version
[0.6.2](https://github.com/jbeder/yaml-cpp/releases/tag/yaml-cpp-0.6.2) is
included in-tree.

Comgr depends on Clang and LLD, which should be built as a part of the LLVM
distribution used. The LLVM, Clang, and LLD used must be from the amd-common
branches of their respective repositories on the [RadeonOpenCompute
Github](https://github.com/RadeonOpenCompute) organisation.

An example command-line to build Comgr on Linux is:

    $ cd ~/llvm/
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=dist \
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ..
    $ make
    $ make install
    $ # build AMDDeviceLibs
    $ cd ~/support/lib/comgr
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="~/llvm/build/dist;path/to/device-libs" ..
    $ make
    $ make test

The equivalent on Windows will use another build tool, such as msbuild or
Visual Studio:

    $ cd "%HOMEPATH%\llvm"
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=dist \
        -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ..
    $ msbuild ALL_BUILD.vcxproj
    $ msbuild INSTALL.vcxproj
    $ rem build AMDDeviceLibs
    $ cd "%HOMEPATH%\support\lib\comgr"
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="%HOMEPATH%\llvm\build\dist;path\to\device-libs" ..
    $ msbuild ALL_BUILD.vcxproj
    $ msbuild RUN_TESTS.vcxproj

Depend on the Code Object Manager
---------------------------------

Comgr exports a CMake package named `amd_comgr` for both the build and install
tree. This package defines a library target named `amd_comgr`. To depend on
this in your CMake project, use `find_package`:

    find_package(amd_comgr REQUIRED CONFIG)
    ...
    target_link_libraries(your_target amd_comgr)

If Comgr is not installed to a standard CMake search directory, the path to the
build or install tree can be supplied to CMake via `CMAKE_PREFIX_PATH`:

    cmake -DCMAKE_PREFIX_PATH=path/to/comgr/build/or/install
