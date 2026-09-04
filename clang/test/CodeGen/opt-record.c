// RUN: %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o %t -opt-record-file %t.yaml -emit-obj
// RUN: cat %t.yaml | FileCheck %s
// RUN: llvm-profdata merge %S/Inputs/opt-record.proftext -o %t.profdata
// RUN: %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata %s -o %t -opt-record-file %t.yaml -emit-obj
// RUN: cat %t.yaml | FileCheck -check-prefix=CHECK -check-prefix=CHECK-PGO %s
// RUN: %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o %t -opt-record-file %t.yaml -opt-record-passes inline -emit-obj
// RUN: cat %t.yaml | FileCheck -check-prefix=CHECK-PASSES %s
// RUN: not %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o %t -opt-record-file %t.yaml -opt-record-passes "(foo" -emit-obj 2>&1 | FileCheck -check-prefix=CHECK-PATTERN-ERROR %s
// RUN: %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o %t -opt-record-file %t.yaml -opt-record-format yaml -emit-obj
// RUN: cat %t.yaml | FileCheck %s
// RUN: not %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o %t -opt-record-file %t.yaml -opt-record-format "unknown-format" -emit-obj 2>&1 | FileCheck -check-prefix=CHECK-FORMAT-ERROR %s
// REQUIRES: x86-registered-target

void bar(void);
void foo(void) { bar(); }

void Test(int *res, int *c, int *d, int *p, int n) {
  int i;

#pragma clang loop vectorize(assume_safety)
  for (i = 0; i < 1600; i++) {
    res[i] = (p[i] == 0) ? res[i] : res[i] + d[i];
  }
}

// CHECK: --- !Missed
// CHECK: Pass:            inline
// CHECK: Name:            NoDefinition
// CHECK: DebugLoc:
// CHECK: Function:        foo
// CHECK-PGO: Hotness:
// CHECK-PASSES: Pass: inline

// CHECK: --- !Passed
// CHECK: Pass:            loop-vectorize
// CHECK: Name:            Vectorized
// CHECK: DebugLoc:
// CHECK: Function:        Test
// CHECK-PGO: Hotness:
// CHECK-PASSES-NOT: loop-vectorize

// CHECK-PATTERN-ERROR: error: in pattern '(foo': parentheses not balanced

// CHECK-FORMAT-ERROR: error: unknown remark serializer format: 'unknown-format'
