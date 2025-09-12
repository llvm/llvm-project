// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

int nested_switch(int a) {
  switch (int b = 1; a) {
  case 0:
    b = b + 1;
  case 1:
    return b;
  case 2: {
    b = b + 1;
    if (a > 1000) {
        case 9:
          b += a;
    }
    if (a > 500) {
        case 7:
          return a + b;
    }
    break;
  }
  }

  return 0;
}

// CHECK: define {{.*}}@_Z13nested_switchi(
// CHECK: switch i32 %6, label %[[DEFAULT_BB:[0-9]+]] [
// CHECK:   i32 0, label %[[ZERO_BB:[0-9]+]]
// CHECK:   i32 1, label %[[ONE_BB:[0-9]+]]
// CHECK:   i32 2, label %[[TWO_BB:[0-9]+]]
// CHECK:   i32 9, label %[[NINE_BB:[0-9]+]]
// CHECK:   i32 7, label %[[SEVEN_BB:[0-9]+]]
// CHECK: ]
//
// CHECK: [[ZERO_BB]]:
// CHECK:   add {{.*}}, 1
// CHECK:   br label %[[ONE_BB]]
//
// CHECK: [[ONE_BB]]:
// CHECK:   ret
//
// CHECK: [[TWO_BB]]:
// CHECK:   add {{.*}}, 1
// CHECK:   br label %[[IF_BB:[0-9]+]]
//
// CHECK: [[IF_BB]]:
// CHECK:   %[[CMP:.+]] = icmp sgt i32 %{{.*}}, 1000
// CHECK:   br i1 %[[CMP]], label %[[IF_TRUE_BB:[0-9]+]], label %[[IF_FALSE_BB:[0-9]+]]
//
// CHECK: [[IF_TRUE_BB]]:
// CHECK:   br label %[[NINE_BB]]
//
// CHECK: [[NINE_BB]]:
// CHECK:   %[[A_VALUE:.+]] = load i32
// CHECK:   %[[B_VALUE:.+]] = load i32
// CHECK:   add nsw i32 %[[B_VALUE]], %[[A_VALUE]]
//
// CHECK: %[[CMP2:.+]] = icmp sgt i32 %{{.*}}, 500
// CHECK:   br i1 %[[CMP2]], label %[[IF2_TRUE_BB:[0-9]+]], label %[[IF2_FALSE_BB:[0-9]+]]
//
// CHECK: [[IF2_TRUE_BB]]:
// CHECK:   br label %[[SEVEN_BB]]
//
// CHECK: [[SEVEN_BB]]:
// CHECK:   ret
//
// CHECK: [[DEFAULT_BB]]:
// CHECK:   ret
