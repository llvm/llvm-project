# libclc

libclc is an open source implementation of the library
requirements of the OpenCL C programming language, as specified by the
OpenCL 1.1 Specification. The following sections of the specification
impose library requirements:

  * 6.1: Supported Data Types
  * 6.2.3: Explicit Conversions
  * 6.2.4.2: Reinterpreting Types Using as_type() and as_typen()
  * 6.9: Preprocessor Directives and Macros
  * 6.11: Built-in Functions
  * 9.3: Double Precision Floating-Point
  * 9.4: 64-bit Atomics
  * 9.5: Writing to 3D image memory objects
  * 9.6: Half Precision Floating-Point

libclc is intended to be used with the Clang compiler's OpenCL frontend.

libclc is designed to be portable and extensible. To this end, it provides
generic implementations of most library requirements, allowing the target
to override the generic implementation at the granularity of individual
functions.

libclc currently supports PTX, AMDGPU, SPIRV and CLSPV targets, but support for
more targets is welcome.

## Configure, build, and install

libclc is built as part of an LLVM runtimes build.

Select the targets to build with `LLVM_RUNTIME_TARGETS`, and enable libclc for
each selected target with the matching
`RUNTIMES_<target-triple>_LLVM_ENABLE_RUNTIMES` cache entry.

#### Configure for the AMDGPU target
```
cd llvm-project
mkdir build
cd build
cmake ../llvm -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release \
  -DRUNTIMES_amdgcn-amd-amdhsa-llvm_LLVM_ENABLE_RUNTIMES=libclc \
  -DLLVM_RUNTIME_TARGETS="amdgcn-amd-amdhsa-llvm"
```

#### Configure for the NVPTX64 target
```
cmake ../llvm -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release \
  -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libclc \
  -DLLVM_RUNTIME_TARGETS="nvptx64-nvidia-cuda"
```

#### Configure for CLSPV targets
```
cmake ../llvm -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release \
  -DRUNTIMES_clspv--_LLVM_ENABLE_RUNTIMES=libclc \
  -DRUNTIMES_clspv64--_LLVM_ENABLE_RUNTIMES=libclc \
  -DLLVM_RUNTIME_TARGETS="clspv--;clspv64--"
```

#### Configure for SPIR-V targets
```
cmake ../llvm -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release \
  -DRUNTIMES_spirv-mesa3d-_LLVM_ENABLE_RUNTIMES=libclc \
  -DRUNTIMES_spirv64-mesa3d-_LLVM_ENABLE_RUNTIMES=libclc \
  -DLLVM_RUNTIME_TARGETS="spirv-mesa3d-;spirv64-mesa3d-"
```

To build multiple targets, pass them as a semicolon-separated list in
`LLVM_RUNTIME_TARGETS` and provide a matching
`RUNTIMES_<target-triple>_LLVM_ENABLE_RUNTIMES=libclc` entry for each target.

#### Build
```
ninja
```

#### Install
```
ninja install
```

Note you can use the `DESTDIR` Makefile variable to do staged installs.
```
DESTDIR=/path/for/staged/install ninja install
```

## Testing
libclc utilizes the LLVM testing infrastructure.
#### Run all tests
To execute all per-target tests for libclc.
```
ninja check-libclc
```
`check-libclc` is a top-level target that aggregates all per-target tests.

#### Run target-specific tests
If you are working on a specific target, you can run tests for just that target triple:
```
ninja check-libclc-<target-triple>
```
Alternatively, you can run target-specific tests via the runtimes build by
pointing to the target-specific build directory:
```
ninja -C runtimes/runtimes-<target-triple>-bins check-libclc
```

## Out-of-tree build

To build out of tree, or in other words, against an existing LLVM build or install:
```
CC=$(<path-to>/llvm-config --bindir)/clang cmake \
  <path-to>/llvm-project/libclc/CMakeLists.txt -DCMAKE_BUILD_TYPE=Release \
  -G Ninja -DLLVM_DIR=$(<path-to>/llvm-config --cmakedir) \
  -DLLVM_RUNTIMES_TARGET=<target-triple>
$ ninja
```
Then install as before.

In both cases, the LLVM used must include the targets you want libclc support for
(`AMDGPU` and `NVPTX` are enabled in LLVM by default). Apart from `SPIRV` where you do
not need an LLVM target but you do need the
[llvm-spirv tool](https://github.com/KhronosGroup/SPIRV-LLVM-Translator) available.
Either build this in-tree, or place it in the directory pointed to by
`LLVM_TOOLS_BINARY_DIR`.

## Website

https://libclc.llvm.org/
