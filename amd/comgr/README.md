Code Object Manager (Comgr)
===========================

The Comgr library provides APIs for compiling and inspecting AMDGPU code
objects. The API is documented in the [header file](include/amd_comgr.h).

Build the Code Object Manager
-----------------------------

`libcomgr.so` contains llvm, and yaml-cpp. The yaml-cpp library version
[0.6.2](https://github.com/jbeder/yaml-cpp/releases/tag/yaml-cpp-0.6.2) is
included in-tree, but llvm must be explicitly specified using the following
CMake variable:

* LLVM_DIR: This should point to the root of the installed LLVM distribution.

Comgr depends on Clang and LLD, which should be built as a part of the LLVM
distribution used.

An example command-line to build Comgr on Linux is:

  $ cd ~/llvm/
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=dist \
      -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ..
  $ make
  $ make install
  $ cd ~/support/lib/comgr
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=~/llvm/build/dist ..
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
  $ cd "%HOMEPATH%\support\lib\comgr"
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="%HOMEPATH%\llvm\build\dist" ..
  $ msbuild ALL_BUILD.vcxproj
  $ msbuild RUN_TESTS.vcxproj
