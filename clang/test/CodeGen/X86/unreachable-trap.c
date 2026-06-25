// RUN: %clang_cc1 %s -O1 -triple=x86_64-unknown-linux-gnu -ftrap-unreachable=all -S  -o - 2>&1 | FileCheck %s --check-prefixes=TRAP,COMMON
// RUN: %clang_cc1 %s -O1 -triple=x86_64-unknown-linux-gnu -ftrap-unreachable=except-noreturn -S  -o - 2>&1 | FileCheck %s --check-prefixes=NORETURN,COMMON
// RUN: %clang_cc1 %s -O1 -triple=x86_64-unknown-linux-gnu -ftrap-unreachable=none -S  -o - 2>&1 | FileCheck %s --check-prefixes=NOTRAP,COMMON

// NOTRAP-NOT: ud2

[[noreturn]]
void exit(int);

#define NULL 0

[[gnu::noinline]]
[[noreturn]]
void a() {
// COMMON-LABEL: a:
// TRAP: ud2
// NORETURN: ud2
    int *ptr = NULL;
    *ptr = 0;
    exit(0);
}

[[gnu::noinline]]
[[noreturn]]
 void b() {
// COMMON-LABEL: b:
// COMMON: call{{.*}} exit
// TRAP: ud2
// NORETURN-NOT: ud2
    exit(0);
}
