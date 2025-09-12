// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

int literals() {
    char a = 'a'; // char literals have char type in C++
    // CHECK:  %{{[0-9]+}} = cir.const #cir.int<97> : !s8i

    return 0;
}
