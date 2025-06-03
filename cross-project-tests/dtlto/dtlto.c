// REQUIRES: x86-registered-target,ld.lld

/// Simple test that DTLTO works with a single input bitcode file and that
/// --save-temps can be applied to the remote compilation.
// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: %clang --target=x86_64-linux-gnu -c -flto=thin %s

// RUN: ld.lld dtlto.o \
// RUN:   --thinlto-distributor=%python \
// RUN:   --thinlto-distributor-arg=%llvm_src_root/utils/dtlto/local.py \
// RUN:   --thinlto-remote-compiler=%clang \
// RUN:   --thinlto-remote-compiler-arg=--save-temps

/// Check that the required output files have been created.
// RUN: ls | count 10
// RUN: ls | FileCheck %s

/// Produced by the bitcode compilation.
// CHECK-DAG: {{^}}dtlto.o{{$}}

/// Linked ELF.
// CHECK-DAG: {{^}}a.out{{$}}

/// --save-temps output for the backend compilation.
// CHECK-DAG: {{^}}dtlto.s{{$}}
// CHECK-DAG: {{^}}dtlto.s.0.preopt.bc{{$}}
// CHECK-DAG: {{^}}dtlto.s.1.promote.bc{{$}}
// CHECK-DAG: {{^}}dtlto.s.2.internalize.bc{{$}}
// CHECK-DAG: {{^}}dtlto.s.3.import.bc{{$}}
// CHECK-DAG: {{^}}dtlto.s.4.opt.bc{{$}}
// CHECK-DAG: {{^}}dtlto.s.5.precodegen.bc{{$}}
// CHECK-DAG: {{^}}dtlto.s.resolution.txt{{$}}

int _start() { return 0; }
