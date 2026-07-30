// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s

int foo() {
    int x = x + 1;
    return x;
}
