// RUN: %clang_cc1 -fsyntax-only %s 2> %t
// RUN: FileCheck < %t %s
#define F (1 << 99)

#define M \
F | F

int a = M;
// CHECK: :8:9: warning: shift count >= width of type [-Wshift-count-overflow]
// CHECK-NEXT:     8 | int a = M;
// CHECK-NEXT:       |         ^
// CHECK-NEXT: :5:11: note: expanded from macro 'M'
// CHECK-NEXT:     5 | #define M \
// CHECK-NEXT:       |           ^
// CHECK-NEXT: :3:14: note: expanded from macro '\
// CHECK-NEXT: F'
// CHECK-NEXT:     3 | #define F (1 << 99)
// CHECK-NEXT:       |              ^  ~~
// CHECK-NEXT: :8:9: warning: shift count >= width of type [-Wshift-count-overflow]
// CHECK-NEXT:     8 | int a = M;
// CHECK-NEXT:       |         ^
// CHECK-NEXT: :6:5: note: expanded from macro 'M'
// CHECK-NEXT:     6 | F | F
// CHECK-NEXT:       |     ^
// CHECK-NEXT: :3:14: note: expanded from macro 'F'
// CHECK-NEXT:     3 | #define F (1 << 99)
// CHECK-NEXT:       |              ^  ~~
