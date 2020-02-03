// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:   | FileCheck %s

// PR14638; make sure this doesn't crash.
struct A {
    bool m_sorted : 1;
};
void func1(bool b, A& a1)
{
    if ((a1.m_sorted = b)) {}
}
// CHECK-LABEL: define void @_Z5func1bR1A
// CHECK: br i1
// CHECK: ret void
