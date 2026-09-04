// RUN: %clang_cc1 -triple x86_64-pc-win32 -ast-dump=json %s | FileCheck %s

template <class>
struct A;

template <class T>
struct A<int T::*> {
    static constexpr int value = sizeof(T);
    static constexpr int array[sizeof(T)];
};

// CHECK-NOT: "mangledName"
