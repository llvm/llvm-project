// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

struct A
{
    ~A();
};
int foo(A);

void bar(A &a)
{
    // CHECK: call void asm
    asm("" : : "r"(foo(a)) ); // rdar://8540491
    // CHECK: call void @_ZN1AD1Ev
}
