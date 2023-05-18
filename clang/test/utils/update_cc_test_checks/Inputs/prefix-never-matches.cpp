// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -O0 -o - %s | FileCheck %s -check-prefix=A
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -O3 -o - %s | FileCheck %s -check-prefix=A

int foo(int i ) {
    return 1;
}