// RUN: %clang_cc1 -fsyntax-only -verify %s 

void f1() {
  struct X {
    struct Y;
  };
  
  struct X::Y {
    void f() {}
  };
}

void f2() {
  struct X {
    struct Y;
    
    struct Y {
      void f() {}
    };
  };
}

// A class nested within a local class is a local class.
void f3(int a) { // expected-note{{'a' declared here}}
  struct X {
    struct Y {
      int f() { return a; } // expected-error{{reference to local variable 'a' declared in enclosing function 'f3'}}
    };
  };
}
