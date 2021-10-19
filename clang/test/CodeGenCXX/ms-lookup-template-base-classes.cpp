// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -fms-compatibility %s -o -
// CHECK that we don't crash.

struct Base {
  void b(int, int);
};

template <typename Base> struct Derived : Base {
  void d() { b(1, 2); }
};

void use() {
  Derived<Base> d;
  d.d();
}
