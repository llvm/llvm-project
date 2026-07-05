// RUN: %clang_pgogen -mllvm -pgo-block-coverage %s -o %t.out
// RUN: env LLVM_PROFILE_FILE=%t1.profraw %run %t.out 1
// RUN: env LLVM_PROFILE_FILE=%t2.profraw %run %t.out 2
// RUN: llvm-profdata merge -o %t.profdata %t1.profraw %t2.profraw
// RUN: %clang_profuse=%t.profdata -mllvm -pgo-verify-bfi -o - -S -emit-llvm %s 2>%t.errs | FileCheck %s --implicit-check-not="!prof"
// RUN: FileCheck %s < %t.errs --allow-empty --check-prefix=CHECK-ERROR

// RUN: llvm-profdata merge -o %t2.profdata %t1.profraw %t1.profraw %t2.profraw %t2.profraw
// RUN: llvm-profdata show %t2.profdata | FileCheck %s --check-prefix=COUNTS

#include <stdlib.h>

// CHECK: @foo({{.*}})
// CHECK-SAME: !prof ![[PROF0:[0-9]+]]
void foo(int a) {
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PROF1:[0-9]+]]
  if (a % 2 == 0) {
    //
  } else {
    //
  }

  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PROF1]]
  for (int i = 1; i < a; i++) {
    // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PROF2:[0-9]+]]
    if (a % 3 == 0) {
      //
    } else {
      // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PROF2]]
      if (a % 1001 == 0) {
        return;
      }
    }
  }

  return;
}

// CHECK: @main({{.*}})
// CHECK-SAME: !prof ![[PROF0]]
int main(int argc, char *argv[]) {
  foo(atoi(argv[1]));
  return 0;
}

// CHECK-DAG: ![[PROF0]] = !{!"function_entry_count", i64 10000}
// CHECK-DAG: ![[PROF1]] = !{!"branch_weights", i32 1, i32 1}
// CHECK-DAG: ![[PROF2]] = !{!"branch_weights", i32 0, i32 1}

// CHECK-ERROR-NOT: warning: {{.*}}: Found inconsistent block coverage

// COUNTS: Maximum function count: 4
