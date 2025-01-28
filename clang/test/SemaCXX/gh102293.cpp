// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

template <typename T> static void destroy() {
    T t;
    ++t;
}

struct Incomplete;

template <typename = int> struct HasD {
  ~HasD() { destroy<Incomplete*>(); }
};

struct HasVT {
  virtual ~HasVT();
};

struct S : HasVT {
  HasD<> v;
};

// Ensure we don't get infinite recursion from the check, however. See GH104802
namespace GH104802 {
class foo {       // expected-note {{definition of 'GH104802::foo' is not complete until the closing '}'}}
  foo a;          // expected-error {{field has incomplete type 'foo'}}

  virtual int c();
};

class bar {       // expected-note {{definition of 'GH104802::bar' is not complete until the closing '}'}}
  const bar a;    // expected-error {{field has incomplete type 'const bar'}}

  virtual int c();
};

class baz {       // expected-note {{definition of 'GH104802::baz' is not complete until the closing '}'}}
  typedef class baz blech;
  blech a;        // expected-error {{field has incomplete type 'blech' (aka 'GH104802::baz')}}

  virtual int c();
};

class quux : quux { // expected-error {{base class has incomplete type}} \
                     expected-note {{definition of 'GH104802::quux' is not complete until the closing '}'}}
  virtual int c();
};
}
