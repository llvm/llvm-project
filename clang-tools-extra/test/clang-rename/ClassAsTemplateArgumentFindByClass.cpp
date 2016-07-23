// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=136 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {};   // CHECK: class Bar {};

template <typename T>
void func() {}

template <typename T>
class Baz {};

int main() {
  func<Foo>();  // CHECK: func<Bar>();
  Baz<Foo> obj; // CHECK: Baz<Bar> obj;
  return 0;
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
