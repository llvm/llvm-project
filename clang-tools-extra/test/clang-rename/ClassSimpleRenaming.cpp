// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=136 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {};       // CHECK: class Bar

int main() {
  Foo *Pointer = 0; // CHECK: Bar *Pointer = 0;
  return 0;
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of Cla when changing
// this file.
