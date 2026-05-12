// REQUIRES: lld-link

/// Simple test that DTLTO works with a single input bitcode file and that
/// --save-temps can be applied to the remote compilation.

// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: %clang --target=x86_64-pc-windows-msvc -c -flto=thin %s -o dtlto.obj

// RUN: lld-link /subsystem:console /entry:_start dtlto.obj \
// RUN:   -thinlto-distributor:%python \
// RUN:   -thinlto-distributor-arg:%llvm_src_root/utils/dtlto/local.py \
// RUN:   -thinlto-remote-compiler:%clang \
// RUN:   -thinlto-remote-compiler-arg:--save-temps

/// Check that the required output files have been created.
// RUN: ls | sort | FileCheck %s

/// No files are expected before.
// CHECK-NOT: {{.}}

/// Linked ELF.
// CHECK: {{^}}dtlto.exe{{$}}

/// Produced by the bitcode compilation.
// CHECK-NEXT: {{^}}dtlto.obj{{$}}

/// --save-temps output for the backend compilation.
// CHECK-NEXT: {{^}}dtlto.s{{$}}
// CHECK-NEXT: {{^}}dtlto.s.0.preopt.bc{{$}}
// CHECK-NEXT: {{^}}dtlto.s.1.promote.bc{{$}}
// CHECK-NEXT: {{^}}dtlto.s.2.internalize.bc{{$}}
// CHECK-NEXT: {{^}}dtlto.s.3.import.bc{{$}}
// CHECK-NEXT: {{^}}dtlto.s.4.opt.bc{{$}}
// CHECK-NEXT: {{^}}dtlto.s.5.precodegen.bc{{$}}
// CHECK-NEXT: {{^}}dtlto.s.resolution.txt{{$}}

/// No files are expected after.
// CHECK-NOT: {{.}}

int _start() { return 0; }
