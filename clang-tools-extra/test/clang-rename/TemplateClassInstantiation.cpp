template <typename T>
class Foo {               // CHECK: class Bar {
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

// RUN: cat %s > %t-0.cpp
// RUN: clang-rename -offset=29 -new-name=Bar %t-0.cpp -i -- -fno-delayed-template-parsing
// RUN: sed 's,//.*,,' %t-0.cpp | FileCheck %s

// RUN: cat %s > %t-1.cpp
// RUN: clang-rename -offset=311 -new-name=Bar %t-1.cpp -i -- -fno-delayed-template-parsing
// RUN: sed 's,//.*,,' %t-1.cpp | FileCheck %s

// RUN: cat %s > %t-2.cpp
// RUN: clang-rename -offset=445 -new-name=Bar %t-2.cpp -i -- -fno-delayed-template-parsing
// RUN: sed 's,//.*,,' %t-2.cpp | FileCheck %s
