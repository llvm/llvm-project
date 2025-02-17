/// Simple test that DTLTO works with a single input file and generates the
/// expected set of files with --save-temps applied to the linker.
///
/// Note that we also supply --save-temps to the compiler for predictable
/// bitcode file names.

// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: %clang --target=x86_64-linux-gnu %s -shared -flto=thin \
// RUN:   -fthinlto-distributor=%python \
// RUN:   -Xthinlto-distributor=%llvm_src_root/utils/dtlto/local.py \
// RUN:   --save-temps \
// RUN:   -fuse-ld=lld \
// RUN:   -nostdlib \
// RUN:   -Wl,--save-temps \
// RUN:   -Werror

/// Check that the required output files have been created.
// RUN: ls | count 13
// RUN: ls | FileCheck %s

/// Produced by the bitcode compilation.
// CHECK-DAG: {{^}}dtlto.bc{{$}}
// CHECK-DAG: {{^}}dtlto.i{{$}}
// CHECK-DAG: {{^}}dtlto.o{{$}}

/// A jobs description JSON and a summary shard is emitted for DTLTO.
// CHECK-DAG: {{^}}a.[[#]].dist-file.json{{$}}
// CHECK-DAG: {{^}}dtlto.[[#]].[[#]].native.o.thinlto.bc{{$}}

/// The backend compilation produces a native object output file for dtlto.o.
// CHECK-DAG: dtlto.[[#]].[[#]].native.o{{$}}

/// Linked ELF.
// CHECK-DAG: {{^}}a.out{{$}}

/// --save-temps incremental files for a.out.
// CHECK-DAG: {{^}}a.out.lto.dtlto.o{{$}}
// CHECK-DAG: {{^}}a.out.0.0.preopt.bc{{$}}
// CHECK-DAG: {{^}}a.out.0.2.internalize.bc{{$}}
// CHECK-DAG: {{^}}a.out.index.bc{{$}}
// CHECK-DAG: {{^}}a.out.index.dot{{$}}
// CHECK-DAG: {{^}}a.out.resolution.txt{{$}}

int _start() { return 0; }
