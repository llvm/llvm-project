// REQUIRES: ld.lld

/// Simple test that DTLTO works with a single input bitcode file and that
/// --save-temps can be applied to the remote compilation.

// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: %clang --target=x86_64-linux-gnu %s -flto=thin -fuse-ld=lld \
// RUN:   -fthinlto-distributor=%python \
// RUN:   -Xthinlto-distributor=%llvm_src_root/utils/dtlto/local.py \
// RUN:   -Wl,--thinlto-remote-compiler-arg=--save-temps \
// RUN:   -nostdlib -Werror

/// Check that the required output files have been created.
// RUN: ls | sort | FileCheck %s

/// No files are expected before.
// CHECK-NOT: {{.}}

/// Linked ELF.
// CHECK: {{^}}a.out{{$}}

/// --save-temps output for the backend compilation.
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP:[a-zA-Z0-9_]+]].s{{$}}
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP]].s.0.preopt.bc{{$}}
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP]].s.1.promote.bc{{$}}
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP]].s.2.internalize.bc{{$}}
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP]].s.3.import.bc{{$}}
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP]].s.4.opt.bc{{$}}
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP]].s.5.precodegen.bc{{$}}
// CHECK-NEXT: {{^}}ld-dtlto-[[TMP]].s.resolution.txt{{$}}

/// No files are expected after.
// CHECK-NOT: {{.}}

int _start() { return 0; }
