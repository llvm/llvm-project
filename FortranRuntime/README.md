# Fortran Runtime

Fortran Runtime is the runtime library for code emitted by the Flang
compiler (https://flang.llvm.org).

## Getting Started

### Bootstrap Build

```sh
cmake -S <path-to-llvm-project>/llvm -DLLVM_ENABLE_PROJECTS=flang -DLLVM_ENABLE_RUNTIMES=FortranRuntime
```

### Runtime-only build

```sh
cmake -S <path-to-llvm-project>/runtimes -DLLVM_ENABLE_RUNTIMES=FortranRuntime -DCMAKE_Fortran_COMPILER=<path-to-llvm-builddir-or-installprefix>/bin/flang-new
```


