#include "comdat_rename.h"
extern void test(FOO *);
FOO foo;
int main() {
  test(&foo);
  foo.caller(20);
  return 0;
}

// The copy of 'caller' defined in this module -- it has
// 'callee' call remaining.
//
// CHECK-LABEL: define {{.*}}caller{{.*}}
// CHECK: {{.*}} call {{.*}}
// CHECK-NOT: br i1 {{.*}}
// CHECK: br {{.*}}label %[[BB1:.*]], label{{.*}}!prof ![[PD1:[0-9]+]]
// CHECK: {{.*}}[[BB1]]:
// CHECK:![[PD1]] = !{!"branch_weights", i64 0, i64 1}
