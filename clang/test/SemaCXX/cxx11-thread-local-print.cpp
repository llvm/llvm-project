// RUN: %clang_cc1 -std=c++11 -triple=x86_64-linux-gnu -ast-print %s | FileCheck %s

// CHECK: __thread int gnu_tl;
// CHECK: _Thread_local int c11_tl;
// CHECK: thread_local int cxx11_tl;
__thread int gnu_tl;
_Thread_local int c11_tl;
thread_local int cxx11_tl;

// CHECK: void foo() {
// CHECK:     thread_local int cxx11_tl;
// CHECK: }
void foo() {
    thread_local int cxx11_tl;
}
