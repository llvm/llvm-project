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

## Compiling and installing

(in the following instructions you can use `make` or `ninja`)

For an in-tree build, Clang must also be built at the same time:
```
$ cmake <path-to>/llvm-project/llvm/CMakeLists.txt -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_ENABLE_RUNTIMES="libclc" -DCMAKE_BUILD_TYPE=Release -G Ninja
$ ninja
```
Then install:
```
$ ninja install
```
Note you can use the `DESTDIR` Makefile variable to do staged installs.
```
$ DESTDIR=/path/for/staged/install ninja install
```
To build out of tree, or in other words, against an existing LLVM build or install:
```
$ cmake <path-to>/llvm-project/libclc/CMakeLists.txt -DCMAKE_BUILD_TYPE=Release \
  -G Ninja -DLLVM_DIR=$(<path-to>/llvm-config --cmakedir)
$ ninja
```
Then install as before.

In both cases this will include all supported targets. You can choose which
targets are enabled by passing `-DLIBCLC_TARGETS_TO_BUILD` to CMake. The default
is `all`.

In both cases, the LLVM used must include the targets you want libclc support for
(`AMDGPU` and `NVPTX` are enabled in LLVM by default). Apart from `SPIRV` where you do
not need an LLVM target but you do need the
[llvm-spirv tool](https://github.com/KhronosGroup/SPIRV-LLVM-Translator) available.
Either build this in-tree, or place it in the directory pointed to by
`LLVM_TOOLS_BINARY_DIR`.

## Website

https://libclc.llvm.org/
