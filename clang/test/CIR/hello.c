// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s

// just confirm that we don't crash
void foo() {}
