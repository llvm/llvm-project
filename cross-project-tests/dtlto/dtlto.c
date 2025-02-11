/// Simple test that DTLTO works with a single input file and generates the
/// expected set of files with --save-temps applied to the linker.
///
/// Note that we also supply --save-temps to the compiler for predictable
/// bitcode file names.

// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: %clang -target x86_64-linux-gnu %s -shared -flto=thin \
// RUN:   -fthinlto-distributor=%python \
// RUN:   -Xdist %llvm_src_root/utils/dtlto/local.py \
// RUN:   --save-temps \
// RUN:   -fuse-ld=lld \
// RUN:   -nostdlib \
// RUN:   -nostartfiles \
// RUN:   -Wl,--save-temps \
// RUN:   -Werror

/// Check that the required output files have been created.
// RUN: ls | count 13
// RUN: ls | FileCheck %s --check-prefix=BITCODE
// RUN: ls | FileCheck %s --check-prefix=BACKEND
// RUN: ls | FileCheck %s --check-prefix=NATIVE
// RUN: ls | FileCheck %s --check-prefix=LLD

/// Files produced by the bitcode compilation.
// BITCODE: dtlto.bc
// BITCODE: dtlto.i
// BITCODE: dtlto.o

/// The DTLTO backend emits the jobs description JSON and a summary shard.
// BACKEND: a.{{[0-9]+}}.dist-file.json
// BACKEND: dtlto.{{[0-9]+}}.{{[0-9]+}}.native.o.thinlto.bc{{$}}

/// Native object output file for dtlto.o.
// NATIVE: dtlto.{{[0-9]+}}.{{[0-9]+}}.native.o{{$}}
/// linked ELF.
// LLD: a.out{{$}}

/// save-temps incremental files for a.out.
/// TODO: Perhaps we should suppress some of the linker hooks for DTLTO.
// LLD: a.out.0.0.preopt.bc{{$}}
// LLD: a.out.0.2.internalize.bc{{$}}
// LLD: a.out.index.bc{{$}}
// LLD: a.out.index.dot{{$}}
// LLD: a.out.lto.dtlto.o{{$}}
// LLD: a.out.resolution.txt{{$}}

int _start() { return 0; }
