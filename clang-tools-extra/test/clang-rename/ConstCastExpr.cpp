// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=136 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {                         // CHECK: class Bar {
public:
  int getValue() {
    return 0;
  }
};

int main() {
  const Foo *C = new Foo();         // CHECK: const Bar *C = new Bar();
  const_cast<Foo *>(C)->getValue(); // CHECK: const_cast<Bar *>(C)->getValue();
}

// Use grep -FUbo 'Cla' <file> to get the correct offset of foo when changing
// this file.
