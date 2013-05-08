// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-nullptr %t.cpp -- -I %S
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *

#define NULL 0

template <typename T>
class A {
public:
  A(T *p = NULL) {}
  // CHECK: A(T *p = nullptr) {}

  void f() {
    Ptr = NULL;
    // CHECK: Ptr = nullptr;
  }

  T *Ptr;
};

template <typename T>
T *f2(T *a = NULL) {
  // CHECK: T *f2(T *a = nullptr) {
  return a ? a : NULL;
  // CHECK: return a ? a : nullptr;
}
