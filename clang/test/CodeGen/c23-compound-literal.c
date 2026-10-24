// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=LIFETIME

struct S { int a; int b; };

// CHECK-LABEL: define dso_local i32 @test1()
// CHECK: %[[VALUE:.*]] = load i32, ptr @.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test1(void) {
  return (static int){42};
}

// CHECK-LABEL: define dso_local i32 @test2()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 7, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test2(void) {
  return (constexpr int){7};
}

// CHECK-LABEL: define dso_local ptr @test3()
// CHECK: ret ptr @.compoundliteral.1
const int *test3(void) {
  return &(static constexpr int){15};
}

// CHECK-LABEL: define dso_local i32 @test4()
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.2)
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr [[ADDR]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test4(void) {
  return (thread_local static int){2};
}

// CHECK-LABEL: define dso_local ptr @test5()
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.3)
// CHECK-NEXT: ret ptr [[ADDR]]
int *test5(void) {
  return &(thread_local static int){2};
}

// CHECK-LABEL: define dso_local i32 @test6()
// CHECK: store ptr @.compoundliteral.4, ptr %a, align 8
// CHECK-NEXT: %[[BASE:.*]] = load ptr, ptr %a, align 8
// CHECK-NEXT: %[[MEMBER:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[BASE]], i32 0, i32 0
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[MEMBER]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test6(void) {
  struct S *a = &(static struct S){1, 2};
  return a->a;
}

// CHECK-LABEL: define dso_local i64 @test7()
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 @.compoundliteral.5, i64 8, i1 false)
// CHECK-NEXT: %[[VALUE:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[VALUE]]
struct S test7(void) {
  return (static struct S){9, 10};
}

// CHECK-LABEL: define dso_local i64 @test8()
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.6)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 [[ADDR]], i64 8, i1 true)
// CHECK-NEXT: %[[VALUE:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[VALUE]]
struct S test8(void) {
  return (thread_local static volatile struct S){11, 12};
}

// CHECK-LABEL: define dso_local i64 @test9()
// CHECK: %.compoundliteral = alloca %struct.S
// CHECK: store i32 13
// CHECK: store i32 14
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 %.compoundliteral, i64 8, i1 true)
// CHECK-NEXT: %[[VALUE:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[VALUE]]
struct S test9(void) {
  return (register volatile struct S){13, 14};
}

// CHECK-LABEL: define dso_local i32 @test10()
// CHECK: %.compoundliteral = alloca %struct.S
// CHECK: store i32 3
// CHECK: store i32 4
// CHECK: store ptr %.compoundliteral, ptr %a, align 8
// CHECK-NEXT: %[[BASE:.*]] = load ptr, ptr %a, align 8
// CHECK-NEXT: %[[MEMBER:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[BASE]], i32 0, i32 0
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[MEMBER]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test10(void) {
  const struct S *a = &(constexpr struct S){3, 4};
  return a->a;
}

// CHECK-LABEL: define dso_local i32 @test11()
// CHECK: %[[VALUE:.*]] = load i32, ptr @.compoundliteral.7, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test11(void) {
  return (static int[]){5, 6, 7}[0];
}

// CHECK-LABEL: define dso_local i32 @test12()
// CHECK: %.compoundliteral = alloca [3 x i32]
// CHECK: store i32 8, ptr %.compoundliteral
// CHECK: store i32 9
// CHECK: store i32 10
// CHECK: %[[ELEMENT:.*]] = getelementptr inbounds [3 x i32], ptr %.compoundliteral, i64 0, i64 0
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[ELEMENT]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test12(void) {
  return (constexpr int[]){8, 9, 10}[0];
}

// CHECK-LABEL: define dso_local i32 @test13()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 99, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test13(void) {
  return (register constexpr int){99};
}

// CHECK-LABEL: define dso_local i32 @test14()
// CHECK: %a = alloca i32
// CHECK: store i32 16, ptr %a
// CHECK-NEXT: %[[VALUE:.*]] = load atomic i32, ptr %a seq_cst, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test14(void) {
  register _Atomic int a = 16;
  return a;
}

// CHECK-LABEL: define dso_local i32 @test15()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 16, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load atomic i32, ptr %.compoundliteral seq_cst, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test15(void) {
  return (register _Atomic int){16};
}

// CHECK-LABEL: define dso_local i64 @test16()
// CHECK: %a = alloca %struct.S, align 8
// CHECK: store i32 17
// CHECK: store i32 18
// CHECK: %[[VALUE:.*]] = load atomic i64, ptr %a seq_cst, align 8
// CHECK-NEXT: store i64 %[[VALUE]], ptr %retval, align 4
// CHECK-NEXT: %[[RESULT:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[RESULT]]
struct S test16(void) {
  register _Atomic(struct S) a = {(struct S){17, 18}};
  return a;
}

// CHECK-LABEL: define dso_local i64 @test17()
// CHECK: %.compoundliteral = alloca %struct.S, align 8
// CHECK: store i32 17
// CHECK: store i32 18
// CHECK: %[[VALUE:.*]] = load atomic i64, ptr %.compoundliteral seq_cst, align 8
// CHECK-NEXT: store i64 %[[VALUE]], ptr %retval, align 4
// CHECK-NEXT: %[[RESULT:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[RESULT]]
struct S test17(void) {
  return (register _Atomic(struct S)){(struct S){17, 18}};
}

// CHECK-LABEL: define dso_local i32 @test18()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 5, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test18(void) {
  return (int){5};
}

// CHECK-LABEL: define dso_local i32 @test19(i32 noundef %a)
// CHECK: %[[OLD:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: %[[INC:.*]] = add nsw i32 %[[OLD]], 1
// CHECK-NEXT: store i32 %[[INC]], ptr %a.addr, align 4
// CHECK: load ptr, ptr @.compoundliteral.8, align 8
// CHECK: %[[RESULT:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: ret i32 %[[RESULT]]
int test19(int a) {
  (void)(static int (*)[a++]){0};
  return a;
}

// CHECK-LABEL: define dso_local i32 @test20(i32 noundef %a)
// CHECK: %[[OLD:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: %[[INC:.*]] = add nsw i32 %[[OLD]], 1
// CHECK-NEXT: store i32 %[[INC]], ptr %a.addr, align 4
// CHECK: %[[ADDR:.*]] = call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @.compoundliteral.9)
// CHECK-NEXT: load ptr, ptr %[[ADDR]], align 8
// CHECK: %[[RESULT:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: ret i32 %[[RESULT]]
int test20(int a) {
  (void)(thread_local static int (*)[a++]){0};
  return a;
}

int test21(void);

// CHECK-LABEL: define dso_local i32 @test22(
// CHECK: %.compoundliteral = alloca i32
// CHECK: [[CALL:%call]] = call i32 @test21()
// CHECK-NEXT: store i32 [[CALL]], ptr %.compoundliteral
// CHECK-NEXT: %[[BOUND:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int test22(int a[(int){test21()}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @test23(
// CHECK: %.compoundliteral = alloca i32
// CHECK: [[CALL:%call]] = call i32 @test21()
// CHECK-NEXT: store i32 [[CALL]], ptr %.compoundliteral
// CHECK-NEXT: %[[BOUND:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int test23(int a[(register int){test21()}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @test24(
// CHECK: %[[BOUND:.*]] = load volatile i32, ptr @.compoundliteral.10, align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int test24(int a[(static volatile int){13}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @test25(
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.11)
// CHECK-NEXT: %[[BOUND:.*]] = load volatile i32, ptr [[ADDR]], align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int test25(int a[(static thread_local volatile int){14}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @test26(
// CHECK-NOT: call i32 @test21()
// CHECK-NOT: load volatile i32
// CHECK: ret i32 0
int test26(int b(int a[((int){test21()} + (static volatile int){15})])) {
  return 0;
}

// CHECK-LABEL: define dso_local i32 @test27(
// CHECK: %.compoundliteral = alloca i32
// CHECK: [[CALL:%call]] = call i32 @test21()
// CHECK-NEXT: store i32 [[CALL]], ptr %.compoundliteral
// CHECK-NEXT: load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 0
int test27(int (*a(void))[((void)(int){test21()}, 1)]) {
  return 0;
}

int test28(const int *);

// LIFETIME-LABEL: define dso_local i32 @test29(
// LIFETIME: call void @llvm.lifetime.start.p0(ptr %.compoundliteral)
// LIFETIME: call i32 @test28(ptr noundef %.compoundliteral)
// LIFETIME-NOT: @llvm.lifetime.end
// LIFETIME: %[[RESULT:.*]] = load i32, ptr %{{.*}}, align 4
// LIFETIME-NEXT: call void @llvm.lifetime.end.p0(ptr %.compoundliteral)
// LIFETIME-NEXT: ret i32 %[[RESULT]]
// CHECK-LABEL: define dso_local i32 @test29(
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 3, ptr %.compoundliteral
// CHECK-NEXT: %[[CALL:.*]] = call i32 @test28(ptr noundef %.compoundliteral)
// CHECK-NEXT: zext i32 %[[CALL]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int test29(int a[test28(&(constexpr int){3})]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i64 @test30()
// CHECK: %.compoundliteral = alloca %struct.S
// CHECK: store i32 19
// CHECK: store i32 20
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 %.compoundliteral, i64 8, i1 true)
// CHECK-NEXT: %[[VALUE:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[VALUE]]
struct S test30(void) {
  return (volatile struct S){19, 20};
}

// CHECK-LABEL: define dso_local i32 @test31()
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr [[ADDR]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int test31(void) {
  return (static __thread int){21};
}
