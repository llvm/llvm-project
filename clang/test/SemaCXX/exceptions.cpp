// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

struct A; // expected-note 4 {{forward declaration of 'A'}}

struct Abstract { virtual void f() = 0; }; // expected-note {{unimplemented pure virtual method 'f'}}

void trys() {
  try {
  } catch(int i) { // expected-note {{previous definition}}
    int j = i;
    int i; // expected-error {{redefinition of 'i'}}
  } catch(float i) {
  } catch(void v) { // expected-error {{cannot catch incomplete type 'void'}}
  } catch(A a) { // expected-error {{cannot catch incomplete type 'A'}}
  } catch(A *a) { // expected-error {{cannot catch pointer to incomplete type 'A'}}
  } catch(A &a) { // expected-error {{cannot catch reference to incomplete type 'A'}}
  } catch(Abstract) { // expected-error {{variable type 'Abstract' is an abstract class}}
  } catch(...) {
    int j = i; // expected-error {{use of undeclared identifier 'i'}}
  }

  try {
  } catch(...) { // expected-error {{catch-all handler must come last}}
  } catch(int) {
  }
}

void throws() {
  throw;
  throw 0;
  throw throw; // expected-error {{cannot throw object of incomplete type 'void'}}
  throw (A*)0; // expected-error {{cannot throw pointer to object of incomplete type 'A'}}
}

void jumps() {
l1:
  goto l5;
  goto l4; // expected-error {{goto into protected scope}}
  goto l3; // expected-error {{goto into protected scope}}
  goto l2; // expected-error {{goto into protected scope}}
  goto l1;
  try { // expected-note 4 {{jump bypasses initialization of try block}}
  l2:
    goto l5;
    goto l4; // expected-error {{goto into protected scope}}
    goto l3; // expected-error {{goto into protected scope}}
    goto l2;
    goto l1;
  } catch(int) { // expected-note 4 {{jump bypasses initialization of catch block}}
  l3:
    goto l5;
    goto l4; // expected-error {{goto into protected scope}}
    goto l3;
    goto l2; // expected-error {{goto into protected scope}}
    goto l1;
  } catch(...) { // expected-note 4 {{jump bypasses initialization of catch block}}
  l4:
    goto l5;
    goto l4;
    goto l3; // expected-error {{goto into protected scope}}
    goto l2; // expected-error {{goto into protected scope}}
    goto l1;
  }
l5:
  goto l5;
  goto l4; // expected-error {{goto into protected scope}}
  goto l3; // expected-error {{goto into protected scope}}
  goto l2; // expected-error {{goto into protected scope}}
  goto l1;
}

struct BadReturn {
  BadReturn() try {
  } catch(...) {
    // Try to hide
    try {
    } catch(...) {
      {
        if (0)
          return; // expected-error {{return in the catch of a function try block of a constructor is illegal}}
      }
    }
  }
  BadReturn(int);
};

BadReturn::BadReturn(int) try {
} catch(...) {
  // Try to hide
  try {
  } catch(int) {
    return; // expected-error {{return in the catch of a function try block of a constructor is illegal}}
  } catch(...) {
    {
      if (0)
        return; // expected-error {{return in the catch of a function try block of a constructor is illegal}}
    }
  }
}

// Cannot throw an abstract type.
class foo {
public:
  foo() {}
  void bar () {
    throw *this; // expected-error{{cannot throw an object of abstract type 'foo'}}
  }
  virtual void test () = 0; // expected-note{{unimplemented pure virtual method 'test'}}
};

namespace PR6831 {
  namespace NA { struct S; }
  namespace NB { struct S; }
  
  void f() {
    using namespace NA;
    using namespace NB;
    try {
    } catch (int S) { 
    }
  }
}
