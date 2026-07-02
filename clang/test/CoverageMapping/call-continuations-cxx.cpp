// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -dump-coverage-mapping -emit-llvm-only -o - %s | FileCheck %s --check-prefix=MAP

int init_value();

struct C {
  int x;
  C() : x(init_value()) {
    x = 2;
  }
};

int constructor_call(void) {
  C c;
  return c.x;
}

// MAP-LABEL: _Z16constructor_callv:
// MAP: Gap,File 0, [[CTOR_LINE:[0-9]+]]:7 -> [[RET_LINE:[0-9]+]]:3 = #1
// MAP-NEXT: File 0, [[RET_LINE]]:3 -> {{[0-9]+}}:2 = #1

// MAP-LABEL: _ZN1CC2Ev:
// MAP: File 0, [[INIT_LINE:[0-9]+]]:11 -> [[INIT_LINE]]:23 = #0
// MAP-NEXT: File 0, [[INIT_LINE]]:25 -> {{[0-9]+}}:4 = #1
