// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-linux-gnu %s
// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-linux-gnu -fms-layout-compatibility=microsoft %s
// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-windows-gnu %s
// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-pc-windows-gnu -fms-layout-compatibility=itanium %s

struct {
    int a : 24;
    char b : 8;
} __attribute__((gcc_struct)) t1;
_Static_assert(sizeof(t1) == 4, "");

#pragma ms_struct on
struct {
    int a : 24;
    char b : 8;
} __attribute__((gcc_struct)) t2;
_Static_assert(sizeof(t2) == 4, "");
#pragma ms_struct off
