// RUN: %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 %s -o %t -dwarf-column-info -opt-record-file %t.yaml -emit-obj
// RUN: cat %t.yaml | FileCheck %s
// RUN: llvm-profdata merge %S/Inputs/opt-record.proftext -o %t.profdata
// RUN: %clang_cc1 -O3 -triple x86_64-unknown-linux-gnu -target-cpu x86-64 -fprofile-instrument-use-path=%t.profdata %s -o %t -dwarf-column-info -opt-record-file %t.yaml -emit-obj
// RUN: cat %t.yaml | FileCheck -check-prefix=CHECK -check-prefix=CHECK-PGO %s
// REQUIRES: x86-registered-target

void bar();
void foo() { bar(); }

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

// CHECK: --- !Passed
// CHECK: Pass:            loop-vectorize
// CHECK: Name:            Vectorized
// CHECK: DebugLoc:
// CHECK: Function:        Test
// CHECK-PGO: Hotness:

