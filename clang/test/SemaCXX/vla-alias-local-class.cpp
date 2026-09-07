// RUN: %clang_cc1 -std=gnu++20 -verify %s

int foo();

// VLA alias used in normal function scope — no error.
int test0() {
  using X = int[foo()];
  X x;
  return 0;
}

// VLA alias declared inside the local class itself — no error.
int test1() {
  struct S {
    using X = int[foo()];
    S() { X x; }
  };
  return 0;
}

// VLA alias from enclosing scope used as a variable in a local class
int test3() {
  using X = int[foo()]; // expected-note {{type declared here}}
  struct S {
    S() {
      X x; // expected-error {{variably modified type 'X' (aka 'int[f()]') from enclosing scope cannot be used in local class 'S'}}
    }
  };
  return 0;
}

// VLA typedef alias from enclosing scope used as a variable in a local class
int test4() {
  typedef int X[foo()]; // expected-note {{type declared here}}
  struct S {
    S() {
      X x; // expected-error {{variably modified type 'X' (aka 'int[f()]') from enclosing scope cannot be used in local class 'S'}}
    }
  };
  return 0;
}

// VLA alias from enclosing scope used in sizeof inside a local class method
int test5() {
  using X = int[foo()]; // expected-note {{type declared here}}
  struct S {
    int method() {
      return sizeof(X); // expected-error {{variably modified type 'X' (aka 'int[f()]') from enclosing scope cannot be used in local class 'S'}}
    }
  };
  return 0;
}

// VLA alias from enclosing scope used in sizeof in an overriding method
int bar(int&);
struct IFace {
  virtual int a() = 0;
  virtual ~IFace();
};

IFace *bad_sizeof_override() {
  int z = 10;
  using X = int[bar(z)]; // expected-note {{type declared here}}
  struct S : public IFace {
    int a() override {
      return sizeof(X); // expected-error {{variably modified type 'X' (aka 'int[f_param(z)]') from enclosing scope cannot be used in local class 'S'}}
    }
  };
  return new S;
}
