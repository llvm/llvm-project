// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=193 -new-name=X %t.cpp -i -- -frtti
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Base {
  virtual int getValue() const = 0;
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

  dynamic_cast<const Derived &>(Reference).getValue(); // CHECK: dynamic_cast<const X &>
  dynamic_cast<const Derived *>(Pointer)->getValue();  // CHECK: dynamic_cast<const X *>
}

// Use grep -FUbo 'Derived' <file> to get the correct offset of foo when changing
// this file.
