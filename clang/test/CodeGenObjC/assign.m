// RUN: %clang_cc1 -triple x86_64 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

struct s0 {
  int x;
};

@interface C0
@property int x0;
@property _Complex int x1;
@property struct s0 x2;
@end

// Check that we get exactly the message sends we expect, and no more.
//
// CHECK: define void @f0
void f0(C0 *a) {
// CHECK: objc_msgSend
  int l0 = (a.x0 = 1);

// CHECK: objc_msgSend
  _Complex int l1 = (a.x1 = 1);

// CHECK: objc_msgSend
  struct s0 l2 = (a.x2 = (struct s0) { 1 });

// CHECK: objc_msgSend
// CHECK: objc_msgSend
  int l3 = (a.x0 += 1);

// CHECK: objc_msgSend
// CHECK: objc_msgSend
  _Complex int l4 = (a.x1 += 1);

// CHECK-NOT: objc_msgSend
// CHECK: }
}
