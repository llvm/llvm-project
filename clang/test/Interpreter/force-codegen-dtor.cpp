// UNSUPPORTED: system-aix

// RUN: cat %s | clang-repl | FileCheck %s
int *x = new int();
template <class T> struct GuardX { T *&x; GuardX(T *&x) : x(x) {}; ~GuardX(); };
template <class T> GuardX<T>::~GuardX() { delete x; x = nullptr; }

// clang would normally defer codegen for ~GuardX()
// Make sure that RuntimeInterfaceBuilder requests it explicitly
(GuardX(x))

// CHECK-NOT: Symbols not found
// CHECK-NOT: _ZN6GuardXIiED2Ev
