// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -dump-coverage-mapping -emit-llvm -o - %s | FileCheck %s --check-prefixes=MAP,IR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -o - %s | FileCheck %s --check-prefix=NOCC

void f(void);
__attribute__((returns_twice)) int returns_twice(void);

int after_call(void) {
  f();
  return 1;
}

// IR-LABEL: define{{.*}} i32 @after_call(
// IR: call void @f()
// IR-NEXT: load i64, ptr getelementptr inbounds ([2 x i64], ptr @__profc_after_call, i32 0, i32 1)
// IR: ret i32 1

int setjmp_like(void) {
  if (returns_twice() == 0)
    return 1;
  return 2;
}

// IR-LABEL: define{{.*}} i32 @setjmp_like
// IR: call{{.*}}@returns_twice
// IR-NEXT: load i64, ptr getelementptr inbounds ([3 x i64], ptr @__profc_setjmp_like, i32 0, i32 2)
// MAP-LABEL: after_call:
// MAP: Gap,File 0, [[@LINE-19]]:7 -> [[@LINE-18]]:3 = #1
// MAP: File 0, [[@LINE-19]]:3 -> [[@LINE-18]]:2 = #1
// MAP-LABEL: setjmp_like:
// MAP: Branch,File 0, [[@LINE-12]]:7 -> [[@LINE-12]]:27 = #1, (#2 - #1)
// NOCC-LABEL: setjmp_like:
// NOCC: Branch,File 0, [[@LINE-14]]:7 -> [[@LINE-14]]:27 = #1, (#0 - #1)
