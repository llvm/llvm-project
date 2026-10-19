// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -emit-llvm -o - %s | FileCheck %s

bool cond();
void inc();

// CHECK-LABEL: define {{.*}}void @_Z15range_for_breakRA4_i(
// CHECK:       for.cond:
// CHECK:       for.body:
// CHECK-NEXT:    br label %for.end5
// CHECK:       for.end:
// CHECK:       for.end5:
void range_for_break(int (&range)[4]) {
  for (; cond(); inc())
    for (int value : *({ break; &range; })) {}
}

// CHECK-LABEL: define {{.*}}void @_Z18range_for_continueRA4_i(
// CHECK:       for.cond:
// CHECK:       for.body:
// CHECK-NEXT:    br label %for.inc4
// CHECK:       for.inc:
// CHECK:       for.end:
// CHECK:       for.inc4:
// CHECK:       for.end5:
void range_for_continue(int (&range)[4]) {
  for (; cond(); inc())
    for (int value : *({ continue; &range; })) {}
}
