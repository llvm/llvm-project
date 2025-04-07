<!--===- README.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Fortran Runtime (Flang-RT)

Flang-RT is the runtime library for code emitted by the Flang compiler
(https://flang.llvm.org).


## Getting Started

There are two build modes for the Flang-RT. The bootstrap build, also
called the in-tree build, and the runtime-only build, also called the
out-of-tree build.
Not to be confused with the terms
[in-source and out-of-source](https://cmake.org/cmake/help/latest/manual/cmake.1.html#introduction-to-cmake-buildsystems)
builds as defined by CMake. In an in-source build, the source directory and the
build directory are identical, whereas with an out-of-source build the
build artifacts are stored somewhere else, possibly in a subdirectory of the
source directory. LLVM does not support in-source builds.


### Requirements

Requirements:
  * [Same as LLVM](https://llvm.org/docs/GettingStarted.html#requirements).


### Bootstrapping Runtimes Build

The bootstrapping build will first build Clang and Flang, then use these
compilers to compile Flang-RT. CMake will create a secondary build tree
configured to use these just-built compilers. The secondary build will reuse
the same build options (Flags, Debug/Release, ...) as the primary build.
It will also ensure that once built, Flang-RT is found by Flang from either
the build- or install-prefix. To enable, add `flang-rt` to
`LLVM_ENABLE_RUNTIMES`:

```bash
cmake -S <path-to-llvm-project-source>/llvm \
  -GNinja                                   \
  -DLLVM_ENABLE_PROJECTS="clang;flang"      \
  -DLLVM_ENABLE_RUNTIMES=flang-rt           \
  ...
```

It is recommended to enable building OpenMP alongside Flang and Flang-RT
as well. This will build `omp_lib.mod` required to use OpenMP from Fortran.
Building Compiler-RT may also be required, particularly on platforms that do
not provide all C-ABI functionality (such as Windows).

```bash
cmake -S <path-to-llvm-project-source>/llvm     \
  -GNinja                                       \
  -DCMAKE_BUILD_TYPE=Release                    \
  -DLLVM_ENABLE_PROJECTS="clang;flang;openmp"   \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt;flang-rt" \
  ...
```

By default, the enabled runtimes will only be built for the host platform
(`-DLLVM_RUNTIME_TARGETS=default`). To add additional targets to support
cross-compilation via `flang --target=<target-triple>`, add more triples to
`LLVM_RUNTIME_TARGETS`, such as
`-DLLVM_RUNTIME_TARGETS="default;aarch64-linux-gnu"`.

After configuration, build, test, and install the runtime(s) via

```shell
$ ninja flang-rt
$ ninja check-flang-rt
$ ninja install
```


### Standalone Runtimes Build

Instead of building Clang and Flang from scratch, the standalone Runtime build
uses CMake's environment introspection to find a C, C++, and Fortran compiler.
The compiler to be used can be controlled using CMake's standard mechanisms such
as `CMAKE_CXX_COMPILER`, `CMAKE_CXX_COMPILER`, and `CMAKE_Fortran_COMPILER`.
`CMAKE_Fortran_COMPILER` must be `flang` built from the same Git commit as
Flang-RT to ensure they are using the same ABI. The C and C++ compiler
can be any compiler supporting the same ABI.

In addition to the compiler, the build must be able to find LLVM development
tools such as `lit` and `FileCheck` that are not found in an LLVM's install
directory. Use `CMAKE_BINARY_DIR` to point to directory where LLVM has
been built. A simple build configuration might look like the following:

```bash
cmake -S <path-to-llvm-project-source>/runtimes              \
  -GNinja                                                    \
  -DLLVM_BINARY_DIR=<path-to-llvm-builddir>                  \
  -DCMAKE_Fortran_COMPILER=<path-to-llvm-builddir>/bin/flang \
  -DCMAKE_Fortran_COMPILER_WORKS=yes                         \
  -DLLVM_ENABLE_RUNTIMES=flang-rt                            \
  ...
```

The `CMAKE_Fortran_COMPILER_WORKS` parameter must be set because otherwise CMake
will test whether the Fortran compiler can compile and link programs which will
obviously fail without a runtime library available yet.

Building Flang-RT for cross-compilation triple, the target triple can
be selected using `LLVM_DEFAULT_TARGET_TRIPLE` AND `LLVM_RUNTIMES_TARGET`.
Of course, Flang-RT can be built multiple times with different build
configurations, but have to be located manually when using with the Flang
driver using the `-L` option.

After configuration, build, test, and install the runtime via

```shell
$ ninja
$ ninja check-flang-rt
$ ninja install
```


## Configuration Option Reference

Flang-RT has the followign configuration options. This is in
addition to the build options the LLVM_ENABLE_RUNTIMES mechanism and
CMake itself provide.

 * `FLANG_RT_INCLUDE_TESTS` (boolean; default: `ON`)

   When `OFF`, does not add any tests and unittests. The `check-flang-rt`
   build target will do nothing.

 * `FLANG_RUNTIME_F128_MATH_LIB` (default: `""`)

   Determines the implementation of `REAL(16)` math functions. If set to
   `libquadmath`, uses `quadmath.h` and `-lquadmath` typically distributed with
   gcc. If empty, disables `REAL(16)` support. For any other value, introspects
   the compiler for `__float128` or 128-bit `long double` support.
   [More details](docs/Real16MathSupport.md).

 * `FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT` (values: `"CUDA"`,`"OpenMP"`, `""` default: `""`)

   When set to `CUDA`, builds Flang-RT with experimental support for GPU
   accelerators using CUDA. `CMAKE_CUDA_COMPILER` must be set if not
   automatically detected by CMake. `nvcc` as well as `clang` are supported.

   When set to `OpenMP`, builds Flang-RT with experimental support for
   GPU accelerators using OpenMP offloading. Only Clang is supported for
   `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`.

 * `FLANG_RT_INCLUDE_CUF` (bool, default: `OFF`)

   Compiles the `libflang_rt.cuda_<CUDA-version>.a/.so` library. This is
   independent of `FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT=CUDA` and only
   requires a
   [CUDA Toolkit installation](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)
   (no `CMAKE_CUDA_COMPILER`).


### Experimental CUDA Support

With `-DFLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT=CUDA`, the following
additional configuration options become available.

 * `FLANG_RT_LIBCUDACXX_PATH` (path, default: `""`)

   Path to libcu++ package installation.

 * `FLANG_RT_CUDA_RUNTIME_PTX_WITHOUT_GLOBAL_VARS` (boolean, default: `OFF`)

   Do not compile global variables' definitions when producing PTX library.
   Default is `OFF`, meaning global variable definitions are compiled by
   default.


### Experimental OpenMP Offload Support

With `-DFLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT=OpenMP`, the following
additional configuration options become available.

 * `FLANG_RT_DEVICE_ARCHITECTURES` (default: `"all"`)

   A list of device architectures that Flang-RT is going to support.
   If `"all"` uses a pre-defined list of architectures. Same purpose as
   `LIBOMPTARGET_DEVICE_ARCHITECTURES` from liboffload.
