// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -dump-coverage-mapping -emit-llvm-only -o - %s | FileCheck %s --check-prefix=MAP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -o - %s | FileCheck %s --check-prefix=NOCC

void f(void);
__attribute__((returns_twice)) int returns_twice(void);

int after_call(void) {
  f();
  return 1;
}

int setjmp_like(void) {
  if (returns_twice() == 0)
    return 1;
  return 2;
}

// MAP-LABEL: after_call:
// MAP: Gap,File 0, [[CALL_LINE:[0-9]+]]:7 -> [[RET_LINE:[0-9]+]]:3 = #1
// MAP: File 0, [[RET_LINE]]:3 -> [[END_LINE:[0-9]+]]:2 = #1
// MAP-LABEL: setjmp_like:
// MAP: Branch,File 0, [[COND_LINE:[0-9]+]]:7 -> [[COND_LINE]]:27 = #1, (#2 - #1)
// NOCC-LABEL: setjmp_like:
// NOCC: Branch,File 0, [[COND_LINE:[0-9]+]]:7 -> [[COND_LINE]]:27 = #1, (#0 - #1)
