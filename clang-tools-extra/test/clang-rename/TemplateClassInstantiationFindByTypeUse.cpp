// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=703 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

// Currently unsupported test.
// FIXME: clang-rename should be able to rename classes with templates
// correctly.
// XFAIL: *

template <typename T>
class Foo {               // CHECK: class Bar;
public:
  T foo(T arg, T& ref, T* ptr) {
    T value;
    int number = 42;
    value = (T)number;
    value = static_cast<T>(number);
    return value;
  }
  static void foo(T value) {}
  T member;
};

template <typename T>
void func() {
  Foo<T> obj;             // CHECK: Bar<T> obj;
  obj.member = T();
  Foo<T>::foo();          // CHECK: Bar<T>::foo();
}

int main() {
  Foo<int> i;             // CHECK: Bar<int> i;
  i.member = 0;
  Foo<int>::foo(0);       // CHECK: Bar<int>::foo(0);

  Foo<bool> b;            // CHECK: Bar<bool> b;
  b.member = false;
  Foo<bool>::foo(false);  // CHECK: Bar<bool>::foo(false);

  return 0;
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of foo when changing
// this file.
