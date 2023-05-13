// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

int literals(void) {
    char a = 'a'; // char literals are int in C
    // CHECK: %[[RES:[0-9]+]] = cir.const(#cir.int<97> : !s32i) : !s32i
    // CHECK: %{{[0-9]+}} = cir.cast(integral, %[[RES]] : !s32i), !s8i

    return 0;
}
