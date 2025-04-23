// RUN: %clang_cc1 -fsyntax-only -verify=c -Wdefault-const-init %s
// RUN: %clang_cc1 -fsyntax-only -verify=c -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=c %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wc++-compat -Wno-default-const-init %s
// good-no-diagnostics

struct A { int i; };
struct S{ const int i; };              // c-note 2 {{member 'i' declared 'const' here}} \
                                          cxx-note 3 {{default constructor of 'S' is implicitly deleted because field 'i' of const-qualified type 'const int' would not be initialized}}
struct T { struct S s; };              // cxx-note {{default constructor of 'T' is implicitly deleted because field 's' has a deleted default constructor}}
struct U { struct S s; const int j; };
struct V { int i; const struct A a; }; // c-note {{member 'a' declared 'const' here}} \
                                          cxx-note {{default constructor of 'V' is implicitly deleted because field 'a' of const-qualified type 'const struct A' would not be initialized}}

void f() {
  struct S s1; // c-warning {{default initialization of an object of type 'struct S' with const member leaves the object unitialized and is incompatible with C++}} \
                  cxx-error {{call to implicitly-deleted default constructor of 'struct S'}}
  struct S s2 = { 0 };
}
void g() {
  struct T t1; // c-warning {{default initialization of an object of type 'struct T' with const member leaves the object unitialized and is incompatible with C++}} \
                  cxx-error {{call to implicitly-deleted default constructor of 'struct T'}}
  struct T t2 = { { 0 } };
}
void h() {
  struct U u1 = { { 0 } };
  struct U u2 = { { 0 }, 0 };
}
void x() {
  struct V v1; // c-warning {{default initialization of an object of type 'struct V' with const member leaves the object unitialized and is incompatible with C++}} \
                  cxx-error {{call to implicitly-deleted default constructor of 'struct V'}}
  struct V v2 = { 0 };
  struct V v3 = { 0, { 0 } };
}

// Test a tentative definition which does eventually get an initializer.
extern const int i;
const int i = 12;

static const int j; // c-warning {{default initialization of an object of type 'const int' is incompatible with C++}} \
                       cxx-error {{default initialization of an object of const type 'const int'}}
const int k;        // c-warning {{default initialization of an object of type 'const int' is incompatible with C++}} \
                       cxx-error {{default initialization of an object of const type 'const int'}}
const struct S s;   // c-warning {{default initialization of an object of type 'const struct S' is incompatible with C++}} \
                       cxx-error {{call to implicitly-deleted default constructor of 'const struct S'}}

void func() {
  const int a;        // c-warning {{default initialization of an object of type 'const int' leaves the object unitialized and is incompatible with C++}} \
                         cxx-error {{default initialization of an object of const type 'const int'}}
  static const int b; // c-warning {{default initialization of an object of type 'const int' is incompatible with C++}} \
                         cxx-error {{default initialization of an object of const type 'const int'}}
}

