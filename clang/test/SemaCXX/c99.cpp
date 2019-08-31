// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=expected,cxx17 -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=expected,cxx20 -std=c++2a %s

// cxx17-warning@* 0+{{designated initializers are a C++20 extension}}

void f1(int i[static 5]) { // expected-error{{C99}}
}

struct Point { int x; int y; int z[]; }; // expected-warning{{flexible array members are a C99 feature}}

Point p1 = { .x = 17,
             y: 25 }; // expected-warning{{use of GNU old-style field designator extension}}

Point p2 = {
  .x = 17, // expected-warning {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
  25 // expected-note {{first non-designated initializer}}
};

Point p3 = {
  .x = 17, // expected-note {{previous initialization is here}}
  .x = 18, // expected-warning {{initializer overrides prior initialization of this subobject}}
};

Point p4 = {
  .x = 17, // expected-warning {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
  25, // expected-note {{first non-designated initializer}}
  // expected-note@-1 {{previous initialization is here}}
  .y = 18, // expected-warning {{initializer overrides prior initialization of this subobject}}
};

int arr[1] = {[0] = 0}; // expected-warning {{array designators are a C99 extension}}

struct Pt { int x, y; };
struct Rect { Pt tl, br; };
Rect r = {
  .tl.x = 0 // expected-warning {{nested designators are a C99 extension}}
};

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
};
struct S {
  int a;
  NonTrivial b;
};
struct T {
  S s;
};
S f();

T t1 = {
  .s = f()
};

// It's important that we reject this; we would not destroy the existing
// 'NonTrivial' object before overwriting it (and even calling its destructor
// would not necessarily be correct).
T t2 = {
  .s = f(), // expected-note {{previous}}
  .s.b = NonTrivial() // expected-error {{initializer would partially override prior initialization of object of type 'S' with non-trivial destruction}}
  // expected-warning@-1 {{nested}}
};

// FIXME: It might be reasonable to accept this.
T t3 = {
  .s = f(), // expected-note {{previous}}
  .s.a = 0 // expected-error {{initializer would partially override prior initialization of object of type 'S' with non-trivial destruction}}
  // expected-warning@-1 {{nested}}
};
