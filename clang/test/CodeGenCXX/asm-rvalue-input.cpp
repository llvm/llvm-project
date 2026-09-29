// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct S { int a, b; };
consteval S make() { return {1, 2}; }

// CHECK-LABEL: define{{.*}} i64 @_Z14consteval_callv(
// CHECK:         [[A:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[TMP:%.*]], i32 0, i32 0
// CHECK-NEXT:    store i32 1, ptr [[A]], align 4
// CHECK-NEXT:    [[B:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[TMP]], i32 0, i32 1
// CHECK-NEXT:    store i32 2, ptr [[B]], align 4
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
long long consteval_call() {
  long long x;
  asm("" : "=r"(x) : "r"(make()));
  return x;
}

// CHECK-LABEL: define{{.*}} i64 @_Z10paren_initi(
// CHECK:         [[A:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[TMP:%.*]], i32 0, i32 0
// CHECK-NEXT:    [[I:%.*]] = load i32, ptr %i.addr, align 4
// CHECK-NEXT:    store i32 [[I]], ptr [[A]], align 4
// CHECK-NEXT:    [[B:%.*]] = getelementptr inbounds nuw %struct.S, ptr [[TMP]], i32 0, i32 1
// CHECK-NEXT:    store i32 2, ptr [[B]], align 4
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
long long paren_init(int i) {
  long long x;
  asm("" : "=r"(x) : "r"(S(i, 2)));
  return x;
}

struct D { int a, b; ~D(); };
D getd();

// CHECK-LABEL: define{{.*}} i64 @_Z9dtor_callv(
// CHECK:         call void @_Z4getdv(ptr {{.*}}sret(%struct.D) align 4 [[TMP:%.*]])
// CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[TMP]], align 4
// CHECK-NEXT:    call i64 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i64 [[V]])
// CHECK:         call void @_ZN1DD1Ev(ptr {{.*}}[[TMP]])
long long dtor_call() {
  long long x;
  asm("" : "=r"(x) : "r"(getd()));
  return x;
}
