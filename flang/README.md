<!--===- README.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# F18

F18 is a ground-up implementation of a Fortran front end written in modern C++.
F18, when combined with LLVM, is intended to replace the Flang compiler.

Flang is a Fortran compiler targeting LLVM.
Visit the [Flang wiki](https://github.com/flang-compiler/flang/wiki)
for more information about Flang.

## Getting Started

Read more about f18 in the [documentation directory](documentation).
Start with the [compiler overview](documentation/Overview.md).

To better understand Fortran as a language
and the specific grammar accepted by f18,
read [Fortran For C Programmers](documentation/FortranForCProgrammers.md)
and
f18's specifications of the [Fortran grammar](documentation/f2018-grammar.txt)
and
the [OpenMP grammar](documentation/OpenMP-4.5-grammar.txt).

Treatment of language extensions is covered
in [this document](documentation/Extensions.md).

To understand the compilers handling of intrinsics,
see the [discussion of intrinsics](documentation/Intrinsics.md).

To understand how an f18 program communicates with libraries at runtime,
see the discussion of [runtime descriptors](documentation/RuntimeDescriptor.md).

If you're interested in contributing to the compiler,
read the [style guide](documentation/C++style.md)
and
also review [how f18 uses modern C++ features](documentation/C++17.md).

## Building F18

### Get the Source Code

```
cd where/you/want/the/source
git clone https://github.com/flang-compiler/f18.git
```

### Supported C++ compilers

F18 is written in C++17.

The code has been compiled and tested with
GCC versions 7.2.0, 7.3.0, 8.1.0, and 8.2.0.

The code has been compiled and tested with
clang version 7.0 and 8.0
using either GNU's libstdc++ or LLVM's libc++.

### LLVM dependency

F18 uses components from LLVM.

The instructions to build LLVM can be found at
https://llvm.org/docs/GettingStarted.html.

We highly recommend using the same compiler to compile both llvm and f18.

The f18 CMakeList.txt file uses
the variable `LLVM_DIR` to find the installed components.

To get the correct LLVM libraries included in your f18 build,
define LLVM_DIR on the cmake command line.
```
LLVM=<LLVM_BUILD_DIR>/lib/cmake/llvm cmake -DLLVM_DIR=$LLVM ...
```
where `LLVM_BUILD_DIR` is
the top-level directory where LLVM was built.

### LLVM dependency when building f18 with Fortran IR

If you do not want to build Fortran IR, add `-DLINK_WITH_FIR=Off` to f18 cmake
command and ignore the rest of this section.

If you intend to build f18 with Fortran IR (`-DLINK_WITH_FIR` On by default),
you must:
- build LLVM with the same compiler and options as the one you are using
to build F18.
- pass `-DCMAKE_CXX_STANDARD=17 -DLLVM_ENABLE_PROJECTS="mlir"`
to LLVM cmake command.
- install LLVM somewhere with `make install` in order to get the required
AddMLIR cmake file (it is not generated in LLVM build directory).

Installing LLVM from packages is most likely not an option as it will not include
MLIR and not be built following C++17 standard.

MLIR is under active development and the most recent development version
may be incompatible. A branch named `f18` is available inside LLVM fork in
https://github.com/flang-compiler/f18-llvm-project. It contains a version of LLVM
that is known be compatible to build f18 with FIR.

The fastest way to get set up is to do:

```
cd where/you/want/to/build/llvm
git clone --depth=1 -b f18 https://github.com/flang-compiler/f18-llvm-project.git
mkdir build
mkdir install
cd build
cmake ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_INSTALL_UTILS=On \
    -DCMAKE_INSTALL_PREFIX=../install
make
make install
```

Then, `-DLLVM_DIR` would have to be set to
 `<where/you/want/to/build/llvm>/install/lib/cmake/llvm` in f18 cmake command.

To run lit tests,
`-DLLVM_EXTERNAL_LIT=<where/you/want/to/build/llvm>/build/bin/llvm-lit` must be
added to f18 cmake command. This is because `llvm-lit` is not part of
LLVM installation.

Note that when using some advanced options from f18 cmake file it may be
necessary to reproduce their effects in LLVM cmake command.

### Building f18 with GCC

By default,
cmake will search for g++ on your PATH.
The g++ version must be one of the supported versions
in order to build f18.

Or,
cmake will use the variable CXX to find the C++ compiler.
CXX should include the full path to the compiler
or a name that will be found on your PATH,
e.g. g++-7.2, assuming g++-7.2 is on your PATH.
```
export CXX=g++-7.2
```
or
```
CXX=/opt/gcc-7.2/bin/g++-7.2 cmake ...
```
There's a third option!
The CMakeList.txt file uses the variable GCC
as the path to the bin directory containing the C++ compiler.

GCC can be defined on the cmake command line
where `<GCC_DIRECTORY>` is the path to a GCC installation with bin, lib, etc:
```
cmake -DGCC=<GCC_DIRECTORY> ...
```

### Building f18 with clang

To build f18 with clang,
cmake needs to know how to find clang++
and the GCC library and tools that were used to build clang++.

The CMakeList.txt file expects either CXX or BUILD_WITH_CLANG to be set.

CXX should include the full path to clang++
or clang++ should be found on your PATH.
```
export CXX=clang++
```
BUILD_WITH_CLANG can be defined on the cmake command line
where `<CLANG_DIRECTORY>`
is the path to a clang installation with bin, lib, etc:
```
cmake -DBUILD_WITH_CLANG=<CLANG_DIRECTORY>
```
Or GCC can be defined on the f18 cmake command line
where `<GCC_DIRECTORY>` is the path to a GCC installation with bin, lib, etc:
```
cmake -DGCC=<GCC_DIRECTORY> ...
```
To use f18 after it is built,
the environment variables PATH and LD_LIBRARY_PATH
must be set to use GCC and its associated libraries.

### Installation Directory

To specify a custom install location,
add
`-DCMAKE_INSTALL_PREFIX=<INSTALL_PREFIX>`
to the cmake command
where `<INSTALL_PREFIX>`
is the path where f18 should be installed.

### Build Types

To create a debug build,
add
`-DCMAKE_BUILD_TYPE=Debug`
to the cmake command.
Debug builds execute slowly.

To create a release build,
add
`-DCMAKE_BUILD_TYPE=Release`
to the cmake command.
Release builds execute quickly.

### Build F18
```
cd ~/f18/build
cmake -DLLVM_DIR=$LLVM ~/f18/src
make
```

### How to Run the Regression Tests

To run all tests:
```
cd ~/f18/build
cmake -DLLVM_DIR=$LLVM ~/f18/src
make check-all
```

To run individual regression tests llvm-lit needs to know the lit
configuration for f18. The parameters in charge of this are:
flang_site_config and flang_config. And they can be set as shown bellow:
```
<path-to-llvm-lit>/llvm-lit \
 --param flang_site_config=<path-to-f18-build>/test-lit/lit.site.cfg.py \
 --param flang_config=<path-to-f18-build>/test-lit/lit.cfg.py \
  <path-to-fortran-test>
```

# How to Generate FIR Documentation

If f18 was built with `-DLINK_WITH_FIR=On` (`On` by default), it is possible to
generate FIR language documentation by running `make flang-doc`. This will
create `docs/Dialect/FIRLangRef.md` in f18 build directory.
