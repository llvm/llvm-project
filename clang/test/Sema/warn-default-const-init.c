// Both of these should enable everything.
// RUN: %clang_cc1 -fsyntax-only -verify=unsafe-var,unsafe-field,zero-init-var,zero-init-field -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=unsafe-var,unsafe-field,zero-init-var,zero-init-field -Wdefault-const-init %s

// This should enable nothing.
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wno-default-const-init-unsafe %s

// Only unsafe field and variable diagnostics
// RUN: %clang_cc1 -fsyntax-only -verify=unsafe-var,unsafe-field %s
// RUN: %clang_cc1 -fsyntax-only -verify=unsafe-var,unsafe-field -Wdefault-const-init-unsafe %s

// Only zero init field and variable diagnostics
// RUN: %clang_cc1 -fsyntax-only -verify=zero-init-var,zero-init-field -Wdefault-const-init -Wno-default-const-init-unsafe %s

// Only zero init and unsafe field diagnostics
// RUN: %clang_cc1 -fsyntax-only -verify=zero-init-field,unsafe-field -Wno-default-const-init-var-unsafe -Wdefault-const-init-field %s

// Only zero init and unsafe variable diagnostics
// RUN: %clang_cc1 -fsyntax-only -verify=zero-init-var,unsafe-var -Wno-default-const-init-field-unsafe -Wdefault-const-init-var %s

// C++ tests
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s

// good-no-diagnostics

struct A { int i; };
struct S{ const int i; };              // unsafe-field-note 2 {{member 'i' declared 'const' here}} \
                                          cxx-note 3 {{default constructor of 'S' is implicitly deleted because field 'i' of const-qualified type 'const int' would not be initialized}}
struct T { struct S s; };              // cxx-note {{default constructor of 'T' is implicitly deleted because field 's' has a deleted default constructor}}
struct U { struct S s; const int j; };
struct V { int i; const struct A a; }; // unsafe-field-note {{member 'a' declared 'const' here}} \
                                          cxx-note {{default constructor of 'V' is implicitly deleted because field 'a' of const-qualified type 'const struct A' would not be initialized}}
struct W { struct A a; const int j; }; // unsafe-field-note {{member 'j' declared 'const' here}} \
                                          cxx-note {{default constructor of 'W' is implicitly deleted because field 'j' of const-qualified type 'const int' would not be initialized}}

void f() {
  struct S s1; // unsafe-field-warning {{default initialization of an object of type 'struct S' with const member leaves the object uninitialized and is incompatible with C++}} \
                  cxx-error {{call to implicitly-deleted default constructor of 'struct S'}}
  struct S s2 = { 0 };
}
void g() {
  struct T t1; // unsafe-field-warning {{default initialization of an object of type 'struct T' with const member leaves the object uninitialized and is incompatible with C++}} \
                  cxx-error {{call to implicitly-deleted default constructor of 'struct T'}}
  struct T t2 = { { 0 } };
}
void h() {
  struct U u1 = { { 0 } };
  struct U u2 = { { 0 }, 0 };
}
void x() {
  struct V v1; // unsafe-field-warning {{default initialization of an object of type 'struct V' with const member leaves the object uninitialized and is incompatible with C++}} \
                  cxx-error {{call to implicitly-deleted default constructor of 'struct V'}}
  struct V v2 = { 0 };
  struct V v3 = { 0, { 0 } };
}
void y() {
  struct W w1; // unsafe-field-warning {{default initialization of an object of type 'struct W' with const member leaves the object uninitialized and is incompatible with C++}} \
                  cxx-error {{call to implicitly-deleted default constructor of 'struct W'}}
  struct W w2 = { 0 };
  struct W w3 = { { 0 }, 0 };
}

// Test a tentative definition which does eventually get an initializer.
extern const int i;
const int i = 12;

static const int j; // zero-init-var-warning {{default initialization of an object of type 'const int' is incompatible with C++}} \
                       cxx-error {{default initialization of an object of const type 'const int'}}
const int k;        // zero-init-var-warning {{default initialization of an object of type 'const int' is incompatible with C++}} \
                       cxx-error {{default initialization of an object of const type 'const int'}}
const struct S s;   // zero-init-var-warning {{default initialization of an object of type 'const struct S' is incompatible with C++}} \
                       cxx-error {{call to implicitly-deleted default constructor of 'const struct S'}}

void func() {
  const int a;        // unsafe-var-warning {{default initialization of an object of type 'const int' leaves the object uninitialized and is incompatible with C++}} \
                         cxx-error {{default initialization of an object of const type 'const int'}}
  static const int b; // zero-init-var-warning {{default initialization of an object of type 'const int' is incompatible with C++}} \
                         cxx-error {{default initialization of an object of const type 'const int'}}
}

