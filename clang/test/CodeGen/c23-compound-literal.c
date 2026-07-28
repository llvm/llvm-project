// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct S { int a; int b; };

// CHECK-LABEL: define dso_local i32 @f1()
// CHECK: %[[VALUE:.*]] = load i32, ptr @.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f1(void) {
  return (static int){42};
}

// CHECK-LABEL: define dso_local i32 @f2()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 7, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f2(void) {
  return (constexpr int){7};
}

// CHECK-LABEL: define dso_local ptr @f3()
// CHECK: ret ptr @.compoundliteral.1
const int *f3(void) {
  return &(static constexpr int){15};
}

// CHECK-LABEL: define dso_local i32 @f4()
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.2)
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr [[ADDR]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f4(void) {
  return (thread_local static int){2};
}

// CHECK-LABEL: define dso_local ptr @f5()
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.3)
// CHECK-NEXT: ret ptr [[ADDR]]
int *f5(void) {
  return &(thread_local static int){2};
}

// CHECK-LABEL: define dso_local i32 @f6()
// CHECK: store ptr @.compoundliteral.4, ptr %a, align 8
// CHECK-NEXT: %[[BASE:.*]] = load ptr, ptr %a, align 8
// CHECK-NEXT: %[[MEMBER:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[BASE]], i32 0, i32 0
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[MEMBER]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f6(void) {
  struct S *a = &(static struct S){1, 2};
  return a->a;
}

// CHECK-LABEL: define dso_local i64 @f7()
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 @.compoundliteral.5, i64 8, i1 true)
// CHECK-NEXT: %[[VALUE:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[VALUE]]
struct S f7(void) {
  return (static volatile struct S){9, 10};
}

// CHECK-LABEL: define dso_local i64 @f8()
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.6)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 [[ADDR]], i64 8, i1 true)
// CHECK-NEXT: %[[VALUE:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[VALUE]]
struct S f8(void) {
  return (thread_local static volatile struct S){11, 12};
}

// CHECK-LABEL: define dso_local i64 @f9()
// CHECK: %.compoundliteral = alloca %struct.S
// CHECK: store i32 13
// CHECK: store i32 14
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %retval, ptr align 4 %.compoundliteral, i64 8, i1 true)
// CHECK-NEXT: %[[VALUE:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[VALUE]]
struct S f9(void) {
  return (register volatile struct S){13, 14};
}

// CHECK-LABEL: define dso_local i32 @f10()
// CHECK: %.compoundliteral = alloca %struct.S
// CHECK: store i32 3
// CHECK: store i32 4
// CHECK: store ptr %.compoundliteral, ptr %a, align 8
// CHECK-NEXT: %[[BASE:.*]] = load ptr, ptr %a, align 8
// CHECK-NEXT: %[[MEMBER:.*]] = getelementptr inbounds nuw %struct.S, ptr %[[BASE]], i32 0, i32 0
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[MEMBER]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f10(void) {
  const struct S *a = &(constexpr struct S){3, 4};
  return a->a;
}

// CHECK-LABEL: define dso_local i32 @f11()
// CHECK: %[[VALUE:.*]] = load i32, ptr @.compoundliteral.7, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f11(void) {
  return (static int[]){5, 6, 7}[0];
}

// CHECK-LABEL: define dso_local i32 @f12()
// CHECK: %.compoundliteral = alloca [3 x i32]
// CHECK: store i32 8, ptr %.compoundliteral
// CHECK: store i32 9
// CHECK: store i32 10
// CHECK: %[[ELEMENT:.*]] = getelementptr inbounds [3 x i32], ptr %.compoundliteral, i64 0, i64 0
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[ELEMENT]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f12(void) {
  return (constexpr int[]){8, 9, 10}[0];
}

// CHECK-LABEL: define dso_local i32 @f13()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 99, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f13(void) {
  return (register constexpr int){99};
}

// CHECK-LABEL: define dso_local i32 @f14()
// CHECK: %a = alloca i32
// CHECK: store i32 16, ptr %a
// CHECK-NEXT: %[[VALUE:.*]] = load atomic i32, ptr %a seq_cst, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f14(void) {
  register _Atomic int a = 16;
  return a;
}

// CHECK-LABEL: define dso_local i32 @f15()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 16, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load atomic i32, ptr %.compoundliteral seq_cst, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f15(void) {
  return (register _Atomic int){16};
}

// CHECK-LABEL: define dso_local i64 @f16()
// CHECK: %a = alloca %struct.S, align 8
// CHECK: store i32 17
// CHECK: store i32 18
// CHECK: %[[VALUE:.*]] = load atomic i64, ptr %a seq_cst, align 8
// CHECK-NEXT: store i64 %[[VALUE]], ptr %retval, align 4
// CHECK-NEXT: %[[RESULT:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[RESULT]]
struct S f16(void) {
  register _Atomic(struct S) a = {(struct S){17, 18}};
  return a;
}

// CHECK-LABEL: define dso_local i64 @f17()
// CHECK: %.compoundliteral = alloca %struct.S, align 8
// CHECK: store i32 17
// CHECK: store i32 18
// CHECK: %[[VALUE:.*]] = load atomic i64, ptr %.compoundliteral seq_cst, align 8
// CHECK-NEXT: store i64 %[[VALUE]], ptr %retval, align 4
// CHECK-NEXT: %[[RESULT:.*]] = load i64, ptr %retval, align 4
// CHECK-NEXT: ret i64 %[[RESULT]]
struct S f17(void) {
  return (register _Atomic(struct S)){(struct S){17, 18}};
}

// CHECK-LABEL: define dso_local i32 @f18()
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 5, ptr %.compoundliteral
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 %[[VALUE]]
int f18(void) {
  return (int){5};
}

// CHECK-LABEL: define dso_local i32 @f19(i32 noundef %a)
// CHECK: %[[OLD:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: %[[INC:.*]] = add nsw i32 %[[OLD]], 1
// CHECK-NEXT: store i32 %[[INC]], ptr %a.addr, align 4
// CHECK: load ptr, ptr @.compoundliteral.8, align 8
// CHECK: %[[RESULT:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: ret i32 %[[RESULT]]
int f19(int a) {
  (void)(static int (*)[a++]){0};
  return a;
}

// CHECK-LABEL: define dso_local i32 @f20(i32 noundef %a)
// CHECK: %[[OLD:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: %[[INC:.*]] = add nsw i32 %[[OLD]], 1
// CHECK-NEXT: store i32 %[[INC]], ptr %a.addr, align 4
// CHECK: %[[ADDR:.*]] = call align 8 ptr @llvm.threadlocal.address.p0(ptr align 8 @.compoundliteral.9)
// CHECK-NEXT: load ptr, ptr %[[ADDR]], align 8
// CHECK: %[[RESULT:.*]] = load i32, ptr %a.addr, align 4
// CHECK-NEXT: ret i32 %[[RESULT]]
int f20(int a) {
  (void)(thread_local static int (*)[a++]){0};
  return a;
}

int f21(void);

// CHECK-LABEL: define dso_local i32 @f22(
// CHECK: %.compoundliteral = alloca i32
// CHECK: [[CALL:%call]] = call i32 @f21()
// CHECK-NEXT: store i32 [[CALL]], ptr %.compoundliteral
// CHECK-NEXT: %[[BOUND:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int f22(int a[(int){f21()}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @f23(
// CHECK: %.compoundliteral = alloca i32
// CHECK: [[CALL:%call]] = call i32 @f21()
// CHECK-NEXT: store i32 [[CALL]], ptr %.compoundliteral
// CHECK-NEXT: %[[BOUND:.*]] = load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int f23(int a[(register int){f21()}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @f24(
// CHECK: %[[BOUND:.*]] = load volatile i32, ptr @.compoundliteral.10, align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int f24(int a[(static volatile int){13}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @f25(
// CHECK: [[ADDR:%.*]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @.compoundliteral.11)
// CHECK-NEXT: %[[BOUND:.*]] = load volatile i32, ptr [[ADDR]], align 4
// CHECK-NEXT: zext i32 %[[BOUND]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int f25(int a[(static thread_local volatile int){14}]) {
  return a[0];
}

// CHECK-LABEL: define dso_local i32 @f26(
// CHECK-NOT: call i32 @f21()
// CHECK-NOT: load volatile i32
// CHECK: ret i32 0
int f26(int b(int a[((int){f21()} + (static volatile int){15})])) {
  return 0;
}

// CHECK-LABEL: define dso_local i32 @f27(
// CHECK: %.compoundliteral = alloca i32
// CHECK: [[CALL:%call]] = call i32 @f21()
// CHECK-NEXT: store i32 [[CALL]], ptr %.compoundliteral
// CHECK-NEXT: load i32, ptr %.compoundliteral, align 4
// CHECK-NEXT: ret i32 0
int f27(int (*a(void))[((void)(int){f21()}, 1)]) {
  return 0;
}

int f28(const int *);

// CHECK-LABEL: define dso_local i32 @f29(
// CHECK: %.compoundliteral = alloca i32
// CHECK: store i32 3, ptr %.compoundliteral
// CHECK-NEXT: %[[CALL:.*]] = call i32 @f28(ptr noundef %.compoundliteral)
// CHECK-NEXT: zext i32 %[[CALL]] to i64
// CHECK: ret i32
// CHECK-NEXT: }
int f29(int a[f28(&(constexpr int){3})]) {
  return a[0];
}
