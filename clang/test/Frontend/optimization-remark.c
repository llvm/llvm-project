// This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
// and -Rpass-analysis) with the inliner. The test is designed to
// always trigger the inliner, so it should be independent of the
// optimization level (under the legacy PM). The inliner is not added to the new
// PM pipeline unless optimizations are present.

// The inliner for the new PM does not seem to be enabled at O0, but we still
// get the same remarks with at least O1. The remarks are also slightly
// different and located in another test file.
// RUN: %clang_cc1 %s -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline -O0 -fno-experimental-new-pass-manager -emit-llvm-only -verify
// RUN: %clang_cc1 %s -Rpass=inline -Rpass-analysis=inline -Rpass-missed=inline -O0 -fno-experimental-new-pass-manager -emit-llvm-only -debug-info-kind=line-tables-only -verify
// RUN: %clang_cc1 %s -Rpass=inline -emit-llvm -o - 2>/dev/null | FileCheck %s
//
// Check that we can override -Rpass= with -Rno-pass.
// RUN: %clang_cc1 %s -Rpass=inline -fno-experimental-new-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
// RUN: %clang_cc1 %s -Rpass=inline -Rno-pass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NO-REMARKS
// RUN: %clang_cc1 %s -Rpass=inline -Rno-everything -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NO-REMARKS
// RUN: %clang_cc1 %s -Rpass=inline -fno-experimental-new-pass-manager -Rno-everything -Reverything -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
//
// The inliner for the new PM does not seem to be enabled at O0, but we still
// get the same remarks with at least O1.
// RUN: %clang_cc1 %s -Rpass=inline -fexperimental-new-pass-manager -O1 -emit-llvm -mllvm -mandatory-inlining-first=false -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
// RUN: %clang_cc1 %s -Rpass=inline -fexperimental-new-pass-manager -O1 -Rno-everything -Reverything -emit-llvm -mllvm -mandatory-inlining-first=false -o -  2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
//
// Check that -w doesn't disable remarks.
// RUN: %clang_cc1 %s -Rpass=inline -fno-experimental-new-pass-manager -w -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
// RUN: %clang_cc1 %s -Rpass=inline -fexperimental-new-pass-manager -O1 -w -emit-llvm -mllvm -mandatory-inlining-first=false -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
//
// FIXME: -Reverything should imply -Rpass=.*.
// RUN: %clang_cc1 %s -Reverything -emit-llvm -o - 2>/dev/null | FileCheck %s --check-prefix=CHECK-NO-REMARKS
//
// FIXME: -Rpass should either imply -Rpass=.* or should be rejected.
// RUN: %clang_cc1 %s -Rpass -emit-llvm -o - 2>/dev/null | FileCheck %s --check-prefix=CHECK-NO-REMARKS

// CHECK-REMARKS: remark:
// CHECK-NO-REMARKS-NOT: remark:

// -Rpass should produce source location annotations, exclusively (just
// like -gmlt).
// CHECK: , !dbg !
// CHECK-NOT: DW_TAG_base_type

// The CU should be marked NoDebug (to prevent writing debug info to
// the final output).
// CHECK: !llvm.dbg.cu = !{![[CU:.*]]}
// CHECK: ![[CU]] = distinct !DICompileUnit({{.*}}emissionKind: NoDebug

int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

float foz(int x, int y) __attribute__((noinline));
float foz(int x, int y) { return x * y; }

// The negative diagnostics are emitted twice because the inliner runs
// twice.
//
int bar(int j) {
// expected-remark@+3 {{foz not inlined into bar because it should never be inlined (cost=never)}}
// expected-remark@+2 {{foz not inlined into bar because it should never be inlined (cost=never)}}
// expected-remark@+1 {{foo inlined into bar}}
  return foo(j, j - 2) * foz(j - 2, j);
}
