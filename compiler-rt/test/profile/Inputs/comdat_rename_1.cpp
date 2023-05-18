#include "comdat_rename.h"
// callee's out-of-line instance profile data -- it comes
// from external calls to it from comdat_rename_2.cpp.
// Its inline instance copy's profile data is different and
// is collected in 'caller''s context. 
int FOO::callee() {
  // CHECK-LABEL: define {{.*}}callee{{.*}}
  // CHECK-NOT: br i1 {{.*}}
  // CHECK: br {{.*}}label{{.*}}, label %[[BB1:.*]], !prof ![[PD1:[0-9]+]]
  // CHECK: {{.*}}[[BB1]]: 
  if (b != 0)
    return a / b;
  if (a != 0)
    return 10 / a;
  return 0;
}

// This is the 'caller''s comdat copy (after renaming) in this module.
// The profile counters include a copy of counters from 'callee':
//
// CHECK-LABEL: define {{.*}}caller{{.*}}
// CHECK-NOT: br i1 {{.*}}
// CHECK: br {{.*}}label{{.*}}, label %[[BB2:.*]], !prof ![[PD2:[0-9]+]]
// CHECK: {{.*}}[[BB2]]: 
// CHECK: br {{.*}}label{{.*}}, label %{{.*}}, !prof !{{.*}}
// CHECK: br {{.*}}label %[[BB3:.*]], label %{{.*}} !prof ![[PD3:[0-9]+]]
// CHECK: {{.*}}[[BB3]]: 
//
// CHECK:![[PD1]] = !{!"branch_weights", i32 0, i32 1}
// CHECK:![[PD2]] = !{!"branch_weights", i32 1, i32 0}
// CHECK:![[PD3]] = !{!"branch_weights", i32 {{.*}}, i32 0}

void test(FOO *foo) { foo->caller(10); }
