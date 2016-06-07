// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=150 -new-name=X %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Base {
};

class Derived : public Base {
public:
  int getValue() const {
    return 0;
  }
};

int main() {
  Derived D;
  const Base &Reference = D;
  const Base *Pointer = &D;

  static_cast<const Derived &>(Reference).getValue(); // CHECK: static_cast<const X &>
  static_cast<const Derived *>(Pointer)->getValue();  // CHECK: static_cast<const X *>
}

// Use grep -FUbo 'Derived' <file> to get the correct offset of foo when changing
// this file.
