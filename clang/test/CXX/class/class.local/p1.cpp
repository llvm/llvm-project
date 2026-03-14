// RUN: %clang_cc1 -fsyntax-only -verify %s 

int x;
void f()
{
  static int s;
  int x; // expected-note{{'x' declared here}}
  extern int g();
  
  struct local {
    int g() { return x; } // expected-error{{reference to local variable 'x' declared in enclosing function 'f'}}
    int h() { return s; }
    int k() { return :: x; }
    int l() { return g(); }
  };
}

local* p = 0; // expected-error{{unknown type name 'local'}}
