// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -dump-coverage-mapping -emit-llvm -o - %s | FileCheck %s --check-prefixes=IR,MAP

int printf(const char *, ...);
void f(void);
int g(void);
__attribute__((returns_twice)) int returns_twice(void);
int tail_callee(int);

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

int musttail_call(int x) {
  __attribute__((musttail)) return tail_callee(x);
}

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

// IR-LABEL: define{{.*}} i32 @musttail_call(
// IR: musttail call i32 @tail_callee
// IR-NEXT: ret i32
// IR-NOT: getelementptr inbounds ({{.*}}@__profc_musttail_call

// MAP-LABEL: block_after_call:
// MAP: Gap,File 0, [[BLOCK_CLOSE:[0-9]+]]:4 -> [[BLOCK_NEXT:[0-9]+]]:3 = #2
// MAP-NEXT: File 0, [[BLOCK_NEXT]]:3 -> [[BLOCK_NEXT]]:18 = #2

// MAP-LABEL: while_call_condition:
// MAP: File 0, [[WHILE_COND:[0-9]+]]:10 -> [[WHILE_COND]]:25 = (#0 + #3)
// MAP-NEXT: Branch,File 0, [[WHILE_COND]]:10 -> [[WHILE_COND]]:25 = #1, (#2 - #1)

// MAP-LABEL: logical_and_call:
// MAP: Branch,File 0, [[LAND_COND:[0-9]+]]:7 -> [[LAND_COND]]:22 = #2, (#4 - #2)
// MAP: Branch,File 0, [[LAND_COND]]:26 -> [[LAND_COND]]:29 = #3, (#5 - #3)

// MAP-LABEL: musttail_call:
// MAP-NEXT: File 0, {{[0-9]+}}:26 -> {{[0-9]+}}:2 = #0
