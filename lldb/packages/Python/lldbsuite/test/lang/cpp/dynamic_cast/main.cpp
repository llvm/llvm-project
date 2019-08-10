#include "ExtBase.h"

class Base {
public:
  virtual char foo() {
    return 'b';
  }
};

class Derived : public Base {
public:
  char foo() override {
    return 'd';
  }
};

class NonOverrideDerived : public Base {
};

class ExtDerived : public ExtBase {
public:
  char bar() override {
    return 'y';
  }
};

int main() {
  Derived d;
  NonOverrideDerived d2;
  Base *b = &d;
  Base *real_base = new Base();
  char c = dynamic_cast<Derived *>(b)->foo();

  ExtDerived ext_d;
  ExtBase *ext_b = &ext_d;
  ExtBase *ext_real_base = new ExtBase();
  c = dynamic_cast<ExtDerived *>(ext_b)->bar();


  return 0; //% self.expect("expression dynamic_cast<class Derived *>(b) == (Derived*)b", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class Base *>(b) == (Base*)b", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class Derived *>(real_base) == nullptr", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class NonOverrideDerived *>(&d) == nullptr", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class ExtDerived *>(real_base) == nullptr", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class Derived *>(&d2) == nullptr", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class NonOverrideDerived *>(&d2) == (NonOverrideDerived *)&d2", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class Derived *>(&ext_d) == nullptr", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class ExtDerived *>(ext_b) == (class ExtDerived*)ext_b", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class ExtBase *>(ext_real_base) == (class ExtBase*)ext_real_base", substrs = ["bool", " = true"])
            //% self.expect("expression dynamic_cast<class ExtDerived *>(ext_real_base) == nullptr", substrs = ["bool", " = true"])
}
