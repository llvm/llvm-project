// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mllvm -enable-single-byte-coverage=true -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -emit-llvm -o - %s | FileCheck %s --check-prefix=SB

int printf(const char *, ...);
void f(void);
int g(void);
__attribute__((returns_twice)) int returns_twice(void);
int tail_callee(int);

int after_call(void) {
  f();
  return 1;
}

int setjmp_like(void) {
  if (returns_twice() == 0)
    return 1;
  return 2;
}

int block_after_call(int argc) {
  {
    if (argc > 2)
      return 2;
    printf("one\n");
  }
  printf("two\n");
  return 0;
}

int while_call_condition(void) {
  while (returns_twice())
    f();
  return 1;
}

int logical_and_call(void) {
  if (returns_twice() && g())
    return 1;
  return 2;
}

int unevaluated_sizeof(void) {
  int x = sizeof(g());
  return x == sizeof(int) ? 0 : 1;
}

int for_increment_call(void) {
  for (int i = 0; i < g(); f())
    g();
  return 0;
}

int musttail_call(int x) {
  __attribute__((musttail)) return tail_callee(x);
}

// IR-DAG: @__profc_unevaluated_sizeof = private global [2 x i64]
// SB-DAG: @__profc_unevaluated_sizeof = private global [3 x i8]

// IR-LABEL: define{{.*}} i32 @after_call(
// IR: call void @f()
// IR-NEXT: load i64, ptr getelementptr inbounds ([2 x i64], ptr @__profc_after_call, i32 0, i32 1)
// IR: ret i32 1

// IR-LABEL: define{{.*}} i32 @setjmp_like(
// IR: call{{.*}} @returns_twice
// IR-NEXT: load i64, ptr getelementptr inbounds ([3 x i64], ptr @__profc_setjmp_like, i32 0, i32 2)

// IR-LABEL: define{{.*}} i32 @block_after_call(
// IR: call{{.*}} @printf(
// IR-NEXT: load i64, ptr getelementptr inbounds ({{.*}}@__profc_block_after_call
// IR: call{{.*}} @printf(
// IR-NEXT: load i64, ptr getelementptr inbounds ({{.*}}@__profc_block_after_call

// IR-LABEL: define{{.*}} i32 @while_call_condition(
// IR: call{{.*}} @returns_twice
// IR-NEXT: load i64, ptr getelementptr inbounds ({{.*}}@__profc_while_call_condition

// IR-LABEL: define{{.*}} i32 @logical_and_call(
// IR: call{{.*}} @returns_twice
// IR-NEXT: load i64, ptr getelementptr inbounds ({{.*}}@__profc_logical_and_call
// IR: call{{.*}} @g
// IR-NEXT: load i64, ptr getelementptr inbounds ({{.*}}@__profc_logical_and_call

// IR-LABEL: define{{.*}} i32 @unevaluated_sizeof(
// IR-NOT: call{{.*}} @g
// IR: ret i32

// IR-LABEL: define{{.*}} i32 @for_increment_call(
// IR: call{{.*}} @f
// IR-NEXT: load i64, ptr getelementptr inbounds ({{.*}}@__profc_for_increment_call

// IR-LABEL: define{{.*}} i32 @musttail_call(
// IR: musttail call i32 @tail_callee
// IR-NEXT: ret i32
// IR-NOT: getelementptr inbounds ({{.*}}@__profc_musttail_call

// SB-LABEL: define{{.*}} i32 @after_call(
// SB: call void @f()
// SB-NEXT: store i8 0, ptr getelementptr inbounds ([2 x i8], ptr @__profc_after_call, i32 0, i32 1)
// SB: ret i32 1

// SB-LABEL: define{{.*}} i32 @setjmp_like(
// SB: call{{.*}} @returns_twice
// SB-NEXT: store i8 0, ptr getelementptr inbounds ([4 x i8], ptr @__profc_setjmp_like, i32 0, i32 2)

// SB-LABEL: define{{.*}} i32 @musttail_call(
// SB: musttail call i32 @tail_callee
// SB-NEXT: ret i32
// SB-NOT: getelementptr inbounds ({{.*}}@__profc_musttail_call
