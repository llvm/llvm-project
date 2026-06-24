# Instrumentor Tools

This directory contains example runtime libraries that demonstrate how to use
the LLVM Instrumentor pass for various profiling and analysis tasks.

## Overview

The LLVM Instrumentor is a configurable instrumentation pass that allows you to
insert runtime calls at various program points (e.g., function entry/exit,
memory operations, floating-point operations). Each example in this directory
provides:

1. A runtime library that implements the instrumentation callbacks
2. An instrumentor configuration JSON file
3. Tests demonstrating usage

## Building

The instrumentor tools are built as part of the compiler-rt build:

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_PROJECTS="clang;compiler-rt"
ninja -C build
```

The runtime libraries will be installed in:
- Darwin: `lib/clang/<version>/lib/darwin/libclang_rt.<example>_osx.a`
- Linux: `lib/clang/<version>/lib/linux/libclang_rt.<example>-<arch>.a`

Configuration files will be installed in `share/llvm/instrumentor-configs/`.

## Adding New Tools 

To add a new instrumentor example:

1. Create a new directory under `compiler-rt/lib/instrumentor-tools/`
2. Add your runtime implementation (`.cpp` and `.h` files)
3. Create an instrumentor configuration JSON file
4. Add a `CMakeLists.txt` (see `flop-counter/CMakeLists.txt` as a template)
5. Update `compiler-rt/lib/instrumentor-tools/CMakeLists.txt` to include your subdirectory
6. Add tests in `compiler-rt/test/instrumentor-tools/`

## Resources

- [Instrumentor Documentation](../../../llvm/docs/Instrumentor.rst)
- [Instrumentor Runtime Headers](../../../llvm/utils/instrumentor_runtime.h)
- [Configuration Wizard](../../../llvm/utils/instrumentor-config-wizard.py)
