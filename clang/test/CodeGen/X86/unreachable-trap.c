// RUN: %clang_cc1 %s -O1 -triple=x86_64-unknown-linux-gnu -ftrap-unreachable -S  -o - 2>&1 | FileCheck %s --check-prefix=TRAP
// RUN: %clang_cc1 %s -O1 -triple=x86_64-unknown-linux-gnu -S  -o - 2>&1 | FileCheck %s --check-prefix=NOTRAP

// TRAP: ud2
// NOTRAP-NOT: ud2

[[noreturn]]
void exit(int);

#define NULL 0

static void test(void) {
    int *ptr = NULL;
    *ptr = 0;
    exit(0);
}

void foo() { test(); }
