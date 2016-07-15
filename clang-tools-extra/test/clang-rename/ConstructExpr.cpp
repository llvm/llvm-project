// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=136 -new-name=Boo %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {};         // CHECK: class Boo {};

int main() {
  Foo *C = new Foo(); // CHECK: Boo *C = new Boo();
}

// Use grep -FUbo 'Boo' <file> to get the correct offset of foo when changing
// this file.
