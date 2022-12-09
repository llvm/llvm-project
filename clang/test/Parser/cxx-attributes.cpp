// RUN: %clang_cc1 -fsyntax-only -verify %s

// GH#58229 - rejects-valid
__attribute__((__visibility__("default"))) [[nodiscard]] int f();
[[nodiscard]] __attribute__((__visibility__("default"))) int f();

class c {
  virtual void f1(const char* a, ...)
    __attribute__ (( __format__(__printf__,2,3) )) = 0;
  virtual void f2(const char* a, ...)
    __attribute__ (( __format__(__printf__,2,3) )) {}
};

template <typename T> class X {
  template <typename S> void X<S>::f() __attribute__((locks_excluded())); // expected-error{{nested name specifier 'X<S>::' for declaration does not refer into a class, class template or class template partial specialization}} \
                                                                          // expected-warning{{attribute locks_excluded ignored, because it is not attached to a declaration}}
};

namespace PR17666 {
  const int A = 1;
  typedef int __attribute__((__aligned__(A))) T1;
  int check1[__alignof__(T1) == 1 ? 1 : -1];

  typedef int __attribute__((aligned(int(1)))) T1;
  typedef int __attribute__((aligned(int))) T2; // expected-error {{expected '(' for function-style cast}}
}

__attribute((typename)) int x; // expected-warning {{unknown attribute 'typename' ignored}}

void fn() {
  void (*__attribute__((attr)) fn_ptr)() = &fn; // expected-warning{{unknown attribute 'attr' ignored}}
  void (*__attribute__((attrA)) *__attribute__((attrB)) fn_ptr_ptr)() = &fn_ptr; // expected-warning{{unknown attribute 'attrA' ignored}} expected-warning{{unknown attribute 'attrB' ignored}}

  void (&__attribute__((attr)) fn_lref)() = fn; // expected-warning{{unknown attribute 'attr' ignored}}
  void (&&__attribute__((attr)) fn_rref)() = fn; // expected-warning{{unknown attribute 'attr' ignored}}

  int i[5];
  int (*__attribute__((attr(i[1]))) pi);  // expected-warning{{unknown attribute 'attr' ignored}}
  pi = &i[0];
}

[[deprecated([""])]] int WrongArgs; // expected-error {{expected variable name or 'this' in lambda capture list}}
[[,,,,,]] int Commas1; // ok
[[,, maybe_unused]] int Commas2; // ok
[[maybe_unused,,,]] int Commas3; // ok
[[,,maybe_unused,]] int Commas4; // ok
[[foo bar]] int NoComma; // expected-error {{expected ','}} \
                         // expected-warning {{unknown attribute 'foo' ignored}}
// expected-error@+2 2 {{expected ']'}}
// expected-error@+1 {{expected external declaration}}
[[foo
