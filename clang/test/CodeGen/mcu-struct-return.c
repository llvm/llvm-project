// RUN: %clang_cc1 -no-opaque-pointers -triple i386-pc-elfiamcu -emit-llvm %s -o - | FileCheck %s

// Structure that is more than 8 byte.
struct Big {
  double a[10];
};

// Empty union with zero size must be returned as void.
union U1 {
} u1;

// Too large union (80 bytes) must be returned via memory.
union U2 {
  struct Big b;
} u2;

// Must be returned in register.
union U3 {
  int x;
} u3;

// Empty struct with zero size, must be returned as void.
struct S1 {
} s1;

// Must be returend in register.
struct S2 {
  int x;
} s2;

// CHECK: [[UNION1_TYPE:%.+]] = type {}
// CHECK: [[UNION2_TYPE:%.+]] = type { [[STRUCT_TYPE:%.+]] }
// CHECK: [[STRUCT_TYPE]] = type { [10 x double] }
// CHECK: [[UNION3_TYPE:%.+]] = type { i32 }
// CHECK: [[STRUCT1_TYPE:%.+]] = type {}
// CHECK: [[STRUCT2_TYPE:%.+]] = type { i32 }

union U1 foo1(void) { return u1; }
union U2 foo2(void) { return u2; }
union U3 foo3(void) { return u3; }
struct S1 bar1(void) { return s1; }
struct S2 bar2(void) { return s2; }
struct S1 bar3(union U1 u) { return s1; }
// CHECK: define{{.*}} void @foo1()
// CHECK: define{{.*}} void @foo2([[UNION2_TYPE]]* noalias sret([[UNION2_TYPE]]) align 4 %{{.+}})
// CHECK: define{{.*}} i32 @foo3()
// CHECK: define{{.*}} void @bar1()
// CHECK: define{{.*}} i32 @bar2()
// CHECK: define{{.*}} void @bar3()

void run(void) {
  union U1 x1 = foo1();
  union U2 x2 = foo2();
  union U3 x3 = foo3();
  struct S1 y1 = bar1();
  struct S2 y2 = bar2();
  struct S1 y3 = bar3(x1);

  // CHECK: [[X1:%.+]] = alloca [[UNION1_TYPE]]
  // CHECK: [[X2:%.+]] = alloca [[UNION2_TYPE]]
  // CHECK: [[X3:%.+]] = alloca [[UNION3_TYPE]]
  // CHECK: [[Y1:%.+]] = alloca [[STRUCT1_TYPE]]
  // CHECK: [[Y2:%.+]] = alloca [[STRUCT2_TYPE]]
  // CHECK: call void @foo1()
  // CHECK: call void @foo2([[UNION2_TYPE]]* sret([[UNION2_TYPE]]) align 4 [[X2]])
  // CHECK: {{.+}} = call i32 @foo3()
  // CHECK: call void @bar1()
  // CHECK: {{.+}} = call i32 @bar2()
  // CHECK: call void @bar3()
}
