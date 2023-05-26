<!--===- docs/GettingStarted.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Getting Started

```eval_rst
.. contents::
   :local:
```

## Building flang
There are two ways to build flang. The first method is to build it at the same
time that you build all of the projects on which it depends. This is called
building in tree. The second method is to first do an in tree build to create
all of the projects on which flang depends.  Then, after creating this base
build, only build the flang code itself. This is called building standalone.
Building standalone has the advantage of being smaller and faster. Once you
create the base build and base install areas, you can create multiple
standalone builds using them.

Note that instructions for building LLVM can be found at
https://llvm.org/docs/GettingStarted.html.

All of the examples below use GCC as the C/C++ compilers and ninja as the build
tool.

### Building flang in tree
Building flang in tree means building flang along with all of the projects on
which it depends.  These projects include mlir, clang, flang, openmp, and
compiler-rt.  Note that compiler-rt is only needed to access libraries that
support 16 bit floating point numbers.  It's not needed to run the automated
tests.  You can use several different C++ compilers for most of the build,
includig GNU and clang.  But building compiler-rt requres using the clang
compiler built in the initial part of the build.

Here's a directory structure that works.  Create a root directory for the
cloned and built files.  Under that root directory, clone the source code
into a directory called llvm-project.  The build will also
create subdirectories under the root directory called build (holds most of
the built files), install (holds the installed files, and compiler-rt (holds
the result of building compiler-rt).

Here's a complete set of commands to clone all of the necessary source and do
the build.

First, create the root directory and `cd` into it.
```bash
mkdir root
cd root

Now clone the source:
```bash
git clone https://github.com/llvm/llvm-project.git
```
Once the clone is complete, execute the following commands:
```bash
rm -rf build
mkdir build
rm -rf install
mkdir install
ROOTDIR=`pwd`
INSTALLDIR=$ROOTDIR/install

cd build

cmake \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$INSTALLDIR \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$LD_LIBRARY_PATH" \
  -DFLANG_ENABLE_WERROR=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_LIT_ARGS=-v \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;flang;openmp" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  ../llvm-project/llvm

ninja
```

By default flang tests that do not specify an explicit `--target` flag use
LLVM's default target triple. For these tests, if there is a need to test on a
different triple by overriding the default, the following needs to be added to
the cmake command above:
`-DLLVM_TARGET_TRIPLE_ENV="<some string>" -DFLANG_TEST_TARGET_TRIPLE="<your triple>"`.

To run the flang tests on this build, execute the command in the "build"
directory:
```bash
ninja check-flang
```

To create the installed files:
```bash
ninja install

echo "latest" > $INSTALLDIR/bin/versionrc
```

To build compiler-rt:
```bash
cd $ROOTDIR
rm -rf compiler-rt
mkdir compiler-rt
cd compiler-rt
CC=$INSTALLDIR/bin/clang \
CXX=$INSTALLDIR/bin/clang++ \
cmake \
  -G Ninja \
  ../llvm-project/compiler-rt \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$INSTALLDIR \
  -DCMAKE_CXX_STANDARD=11 \
  -DCMAKE_C_CFLAGS=-mlong-double-128 \
  -DCMAKE_CXX_CFLAGS=-mlong-double-128 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCOMPILER_RT_BUILD_ORC=OFF \
  -DCOMPILER_RT_BUILD_XRAY=OFF \
  -DCOMPILER_RT_BUILD_MEMPROF=OFF \
  -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
  -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
  -DLLVM_CONFIG_PATH=$INSTALLDIR/bin/llvm-config

ninja
ninja install
```

Note that these instructions specify flang as one of the projects to build in
the in tree build.  This is not strictly necessary for subsequent standalone
builds, but doing so lets you run the flang tests to verify that the source
code is in good shape.

### Building flang standalone
To do the standalone build, start by building flang in tree as described above.
This build is base build for subsequent standalone builds.  Start each
standalone build the same way by cloning the source for llvm-project:
```bash
mkdir standalone
cd standalone
git clone https://github.com/llvm/llvm-project.git
```
Once the clone is complete, execute the following commands:
```bash
cd llvm-project/flang
rm -rf build
mkdir build
cd build

cmake \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$LD_LIBRARY_PATH" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DFLANG_ENABLE_WERROR=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_BUILD_MAIN_SRC_DIR=$ROOTDIR/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$ROOTDIR/build/bin/llvm-lit \
  -DLLVM_LIT_ARGS=-v \
  -DLLVM_DIR=$ROOTDIR/build/lib/cmake/llvm \
  -DCLANG_DIR=$ROOTDIR/build/lib/cmake/clang \
  -DMLIR_DIR=$ROOTDIR/build/lib/cmake/mlir \
  ..

ninja
```

To run the flang tests on this build, execute the command in the "flang/build"
directory:
```bash
ninja check-flang
```

## Supported C++ compilers

Flang is written in C++17.

The code has been compiled and tested with GCC versions from 7.2.0 to 9.3.0.

The code has been compiled and tested with clang version 7.0, 8.0, 9.0 and 10.0
using either GNU's libstdc++ or LLVM's libc++.

The code has been compiled on AArch64, x86_64 and ppc64le servers with CentOS7,
Ubuntu18.04, Rhel, MacOs, Mojave, XCode and Apple Clang version 10.0.1.

Note that flang is not supported on 32 bit CPUs.

### Building flang with GCC

By default,
cmake will search for g++ on your PATH.
The g++ version must be one of the supported versions
in order to build flang.

Or, cmake will use the variable CXX to find the C++ compiler. CXX should include
the full path to the compiler or a name that will be found on your PATH, e.g.
g++-8.3, assuming g++-8.3 is on your PATH.

```bash
export CXX=g++-8.3
```
or
```bash
CXX=/opt/gcc-8.3/bin/g++-8.3 cmake ...
```

### Building flang with clang

To build flang with clang,
cmake needs to know how to find clang++
and the GCC library and tools that were used to build clang++.

CXX should include the full path to clang++
or clang++ should be found on your PATH.
```bash
export CXX=clang++
```

### Installation Directory

To specify a custom install location,
add
`-DCMAKE_INSTALL_PREFIX=<INSTALL_PREFIX>`
to the cmake command
where `<INSTALL_PREFIX>`
is the path where flang should be installed.

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

## How to Run Tests

Flang supports 2 different categories of tests
1. Regression tests (https://www.llvm.org/docs/TestingGuide.html#regression-tests)
2. Unit tests (https://www.llvm.org/docs/TestingGuide.html#unit-tests)

### For standalone builds
To run all tests:
```bash
cd ~/flang/build
cmake -DLLVM_DIR=$LLVM -DMLIR_DIR=$MLIR ~/flang/src
ninja check-all
```

To run individual regression tests llvm-lit needs to know the lit
configuration for flang. The parameters in charge of this are:
flang_site_config and flang_config. And they can be set as shown below:
```bash
<path-to-llvm-lit>/llvm-lit \
 --param flang_site_config=<path-to-flang-build>/test-lit/lit.site.cfg.py \
 --param flang_config=<path-to-flang-build>/test-lit/lit.cfg.py \
  <path-to-fortran-test>

```

Unit tests:

If flang was built with `-DFLANG_INCLUDE_TESTS=ON` (`ON` by default), it is possible to generate unittests.
Note: Unit-tests will be skipped for LLVM install for an standalone build as it does not include googletest related headers and libraries.

There are various ways to run unit-tests.

```

1. ninja check-flang-unit
2. ninja check-all or ninja check-flang
3. <path-to-llvm-lit>/llvm-lit \
        test/Unit
4. Invoking tests from <standalone flang build>/unittests/<respective unit test folder>

```


### For in tree builds
If flang was built with `-DFLANG_INCLUDE_TESTS=ON` (`ON` by default), it is possible to
generate unittests.

To run all of the flang unit tests use the `check-flang-unit` target:
```bash
ninja check-flang-unit
```
To run all of the flang regression tests use the `check-flang` target:
```bash
ninja check-flang
```

## How to Generate Documentation

### Generate FIR Documentation
If flang was built with `-DLINK_WITH_FIR=ON` (`ON` by default), it is possible to
generate FIR language documentation by running `ninja flang-doc`. This will
create `<build-dir>/tools/flang/docs/Dialect/FIRLangRef.md` in flang build directory.

### Generate Doxygen-based Documentation
To generate doxygen-style documentation from source code
- Pass `-DLLVM_ENABLE_DOXYGEN=ON -DFLANG_INCLUDE_DOCS=ON` to the cmake command.

```bash
cd ~/llvm-project/build
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;flang" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_DOXYGEN=ON -DFLANG_INCLUDE_DOCS=ON ../llvm
ninja doxygen-flang
```

It will generate html in

```bash
    <build-dir>/tools/flang/docs/doxygen/html # for flang docs
```
### Generate Sphinx-based Documentation
[Flang documentation](https://flang.llvm.org/docs/) should preferably be written in `markdown(.md)` syntax (they can be in `reStructuredText(.rst)` format as well but markdown is recommended in first place), it
is mostly meant to be processed by the Sphinx documentation generation
system to create HTML pages which would be hosted on the webpage of flang and
updated periodically.

If you would like to generate and view the HTML locally:
- Install [Sphinx](http://sphinx-doc.org/), including the [sphinx-markdown-tables](https://pypi.org/project/sphinx-markdown-tables/) extension.
- Pass `-DLLVM_ENABLE_SPHINX=ON -DSPHINX_WARNINGS_AS_ERRORS=OFF` to the cmake command.

```bash
cd ~/llvm-project/build
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;flang" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_SPHINX=ON -DSPHINX_WARNINGS_AS_ERRORS=OFF ../llvm
ninja docs-flang-html
```

It will generate html in

```bash
   $BROWSER <build-dir>/tools/flang/docs/html/
```

