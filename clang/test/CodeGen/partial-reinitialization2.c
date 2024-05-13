// RUN: %clang_cc1 %s -triple x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s

struct P1 { char x[6]; } g1 = { "foo" };
struct LP1 { struct P1 p1; };

struct P2    { int a, b, c; } g2 = { 1, 2, 3 };
struct LP2   { struct P2 p2; };
struct LP2P2 { struct P2 p1, p2; };
union  UP2   { struct P2 p2; };

struct LP3 { struct P1 p1[2]; } g3 = { { "dog" }, { "cat" } };
struct LLP3 { struct LP3 l3; };
union ULP3 { struct LP3 l3; };

// CHECK-LABEL: test1
void test1(void)
{
  // CHECK: call void @llvm.memcpy{{.*}}ptr align 1 @g1, i64 6, i1 false)
  // CHECK: store i8 120, ptr %

  struct LP1 l = { .p1 = g1, .p1.x[2] = 'x' };
}

// CHECK-LABEL: test2
void test2(void)
{
  // CHECK: call void @llvm.memcpy{{.*}}ptr align 1 @g1, i64 6, i1 false)
  // CHECK: store i8 114, ptr %

  struct LP1 l = { .p1 = g1, .p1.x[1] = 'r' };
}

// CHECK-LABEL: test3
void test3(void)
{
  // CHECK: call void @llvm.memcpy{{.*}}ptr align 4 @g2, i64 12, i1 false)
  // CHECK: store i32 10, ptr %

  struct LP2 l = { .p2 = g2, .p2.b = 10 };
}

// CHECK-LABEL: get235
struct P2 get235(void)
{
  struct P2 p = { 2, 3, 5 };
  return p;
}

// CHECK-LABEL: get456789
struct LP2P2 get456789(void)
{
  struct LP2P2 l = { { 4, 5, 6 }, { 7, 8, 9 } };
  return l;
}

// CHECK-LABEL: get123
union UP2 get123(void)
{
  union UP2 u = { { 1, 2, 3 } };
  return u;
}

// CHECK-LABEL: test4
void test4(void)
{
  // CHECK: [[CALL:%[a-z0-9]+]] = call {{.*}}@get123()
  // CHECK: store{{.*}}[[CALL]], {{.*}}[[TMP0:%[a-z0-9.]+]]
  // CHECK: call void @llvm.memcpy{{.*}}[[TMP0]], i64 12, i1 false)
  // CHECK: store i32 100, ptr %

  struct LUP2 { union UP2 up; } var = { get123(), .up.p2.a = 100 };
}

// CHECK-LABEL: test5
void test5(void)
{
  // .l3 = g3
  // CHECK: call void @llvm.memcpy{{.*}}ptr align 1 @g3, i64 12, i1 false)

  // .l3.p1 = { [0] = g1 } implicitly sets [1] to zero
  // CHECK: call void @llvm.memcpy{{.*}}ptr align 1 @g1, i64 6, i1 false)
  // CHECK: getelementptr{{.*}}%struct.P1, ptr{{.*}}i64 1
  // CHECK: call void @llvm.memset{{.*}}i8 0, i64 6, i1 false)

  // .l3.p1[1].x[1] = 'x'
  // CHECK: store i8 120, ptr %

  struct LLP3 var = { .l3 = g3, .l3.p1 = { [0] = g1 }, .l3.p1[1].x[1] = 'x' };
}

// CHECK-LABEL: test6
void test6(void)
{
  // CHECK: [[LP:%[a-z0-9]+]] = getelementptr{{.*}}%struct.LLP2P2, ptr{{.*}}, i32 0, i32 0
  // CHECK: call {{.*}}get456789(ptr {{.*}}[[LP]])

  // CHECK: [[CALL:%[a-z0-9]+]] = call {{.*}}@get235()
  // CHECK: store{{.*}}[[CALL]], {{.*}}[[TMP0:%[a-z0-9.]+]]
  // CHECK: call void @llvm.memcpy{{.*}}[[TMP0]], i64 12, i1 false)

  // CHECK: store i32 10, ptr %

  struct LLP2P2 { struct LP2P2 lp; } var =  { get456789(),
                                              .lp.p1 = get235(),
                                              .lp.p1.b = 10 };
}
