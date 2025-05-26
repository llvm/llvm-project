# LLVM Project

## Introduction

The LLVM Project is a collection of modular and reusable compiler and toolchain technologies. Despite its name, LLVM has little to do with traditional virtual machines. The name "LLVM" itself is not an acronym; it is the full name of the project.

LLVM is an umbrella project that includes:
- LLVM Core libraries (optimizer, code generators, etc.)
- Clang: A C/C++/Objective-C compiler frontend
- LLDB: A debugger
- lld: A linker
- And many other subprojects

## Prerequisites

To build LLVM, you'll need:

- A C++17 compatible compiler (GCC ≥ 7.1.0, Clang ≥ 5.0.0, Apple Clang ≥ 10.0.0, MSVC ≥ 19.14)
- CMake ≥ 3.13.4
- Python ≥ 3.6
- Git (for version control)
- Ninja build system (recommended) or GNU Make

On macOS, you can install the required tools with:
```bash
brew install cmake ninja python
```

## Building LLVM

### Basic Build Instructions

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   ```

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Configure with CMake:
   ```bash
   cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON
   ```
   
   Note: Adjust the `-DLLVM_ENABLE_PROJECTS` parameter to include only the subprojects you need.

4. Build:
   ```bash
   ninja
   ```

5. (Optional) Install:
   ```bash
   ninja install
   ```

### Using the New Pass Manager

LLVM has transitioned to a new pass manager. When using tools like `opt`, use the new syntax:

```bash
# New pass manager syntax
./bin/opt -passes=instcombine,reassociate -debug-pass-manager -disable-output test.ll

# Old syntax (deprecated)
# ./bin/opt -instcombine -reassociate -debug-pass=List -disable-output test.ll
```

## Documentation

For more detailed information, refer to the official LLVM documentation:

- [LLVM Documentation](https://llvm.org/docs/)
- [Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html)
- [Building LLVM with CMake](https://llvm.org/docs/CMake.html)
- [LLVM Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html)
- [New Pass Manager](https://llvm.org/docs/NewPassManager.html)

## License

LLVM is available under the [Apache License v2.0 with LLVM Exceptions](https://llvm.org/LICENSE.txt).

-----------------------------------------------------------------------------------------------------------

# LLVM with -debug-pass-list Feature (New Pass Manager)

This is a fork of the [official LLVM Project](https://github.com/llvm/llvm-project) focused on implementing an experimental `-debug-pass-list` command-line option for the New Pass Manager (NPM).

## Feature: -debug-pass-list

This feature provides a simple, flat list of pass names invoked by the New Pass Manager for a given optimization level. It's intended to be a more beginner-friendly way to see the sequence of operations compared to more verbose options.

### How to Use
(Assuming you have built this version of Clang/LLVM)

With `clang`:
```bash
./bin/clang -O2 -mllvm -debug-pass-list your_file.c -o /dev/null
# Or to avoid linker issues during testing:
# ./bin/clang -O2 -mllvm -debug-pass-list -S your_file.c -o /dev/null



with opt:
# First generate LLVM IR:
# ./bin/clang -O2 -S -emit-llvm your_file.c -o your_file.ll

# Then run opt:
./bin/opt -passes='default<O2>' -debug-pass-list -disable-output your_file.ll
