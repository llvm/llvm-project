// RUN: %check_clang_tidy %s bugprone-virtual-arithmetic %t

class Base {
public:  
  virtual ~Base() {}
};

class Derived : public Base {};

void operators() {
  Base *b = new Derived[10];

  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on class that declares a virtual function, undefined behavior if the pointee is a different class [bugprone-virtual-arithmetic]

  b = b + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: pointer arithmetic on class that declares a virtual function, undefined behavior if the pointee is a different class [bugprone-virtual-arithmetic]

  b++;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on class that declares a virtual function, undefined behavior if the pointee is a different class [bugprone-virtual-arithmetic]

  b[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on class that declares a virtual function, undefined behavior if the pointee is a different class [bugprone-virtual-arithmetic]

  delete[] static_cast<Derived*>(b);
}

void subclassWarnings() {
  Base *b = new Base[10];

  // False positive that's impossible to distinguish without
  // path-sensitive analysis, but the code is bug-prone regardless.
  b += 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: pointer arithmetic on class that declares a virtual function, undefined behavior if the pointee is a different class [bugprone-virtual-arithmetic]

  delete[] b;

  // Common false positive is a class that overrides all parent functions.
  Derived *d = new Derived[10];

  d += 1;
  // no-warning

  delete[] d;
}