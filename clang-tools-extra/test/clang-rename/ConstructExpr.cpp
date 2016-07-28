// RUN: clang-rename -offset=74 -new-name=Boo %s -- | FileCheck %s

class Foo {};         // CHECK: class Boo {};

int main() {
  Foo *C = new Foo(); // CHECK: Boo *C = new Boo();
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
