// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -ftrap-unreachable -fms-extensions -triple x86_64-windows-msvc -O2 -S %s -o - | FileCheck %s

// CHECK-LABEL: my_noreturn_func:
// CHECK:       movl $0, (%rax)
// CHECK-NEXT:  ud2

extern long volatile *gtrap;

__declspec(noreturn) void my_noreturn_func(void) {
    *gtrap = 0;
}

