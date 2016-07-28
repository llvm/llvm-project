// RUN: clang-rename -offset=74 -new-name=Bar %s -- | FileCheck %s

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

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
