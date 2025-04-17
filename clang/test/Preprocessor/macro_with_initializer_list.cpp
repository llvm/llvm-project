// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -std=c++11 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
namespace std {
  template <class X>
  class initializer_list {
    public: const X *a, *b;
    initializer_list();
  };
}

class Foo {
public:
  Foo();
  Foo(std::initializer_list<int>);
  bool operator==(const Foo);
  Foo operator+(const Foo);
};

#define EQ(x,y) (void)(x == y)  // expected-note 6{{defined here}}
void test_EQ() {
  Foo F;
  F = Foo{1,2};

  EQ(F,F);
  EQ(F,Foo());
  EQ(F,Foo({1,2,3}));
  EQ(Foo({1,2,3}),F);
  EQ((Foo{1,2,3}),(Foo{1,2,3}));
  EQ(F, F + F);
  EQ(F, Foo({1,2,3}) + Foo({1,2,3}));
  EQ(F, (Foo{1,2,3} + Foo{1,2,3}));

  EQ(F,Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  EQ(Foo{1,2,3},F);
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  EQ(Foo{1,2,3},Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}

  EQ(Foo{1,2,3} + Foo{1,2,3}, F);
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  EQ(F, Foo({1,2,3}) + Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  EQ(F, Foo{1,2,3} + Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
}

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{33:8-33:8}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{33:18-33:18}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{36:6-36:6}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{36:16-36:16}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{39:6-39:6}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{39:16-39:16}:")"
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{39:17-39:17}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{39:27-39:27}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{43:6-43:6}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{43:29-43:29}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{46:9-46:9}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{46:34-46:34}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{49:9-49:9}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{49:32-49:32}:")"

#define NE(x,y) (void)(x != y)  // expected-note 6{{defined here}}
// Operator != isn't defined.  This tests that the corrected macro arguments
// are used in the macro expansion.
void test_NE() {
  Foo F;

  NE(F,F);
  // expected-error@-1 {{invalid operands}}
  NE(F,Foo());
  // expected-error@-1 {{invalid operands}}
  NE(F,Foo({1,2,3}));
  // expected-error@-1 {{invalid operands}}
  NE((Foo{1,2,3}),(Foo{1,2,3}));
  // expected-error@-1 {{invalid operands}}

  NE(F,Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  // expected-error@-3 {{invalid operands}}
  NE(Foo{1,2,3},F);
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  // expected-error@-3 {{invalid operands}}
  NE(Foo{1,2,3},Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  // expected-error@-3 {{invalid operands}}

  NE(Foo{1,2,3} + Foo{1,2,3}, F);
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  // expected-error@-3 {{invalid operands}}
  NE(F, Foo({1,2,3}) + Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  // expected-error@-3 {{invalid operands}}
  NE(F, Foo{1,2,3} + Foo{1,2,3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}
  // expected-error@-3 {{invalid operands}}
}

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{89:8-89:8}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{89:18-89:18}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{93:6-93:6}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{93:16-93:16}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{97:6-97:6}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{97:16-97:16}:")"
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{97:17-97:17}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{97:27-97:27}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{102:6-102:6}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{102:29-102:29}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{106:9-106:9}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{106:34-106:34}:")"

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{110:9-110:9}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{110:32-110:32}:")"

#define INIT(var, init) Foo var = init; // expected-note 3{{defined here}}
// Can't use an initializer list as a macro argument.  The commas in the list
// will be interpretted as argument separaters and adding parenthesis will
// make it no longer an initializer list.
void test() {
  INIT(a, Foo());
  INIT(b, Foo({1, 2, 3}));
  INIT(c, Foo());

  INIT(d, Foo{1, 2, 3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-note@-2 {{parentheses are required}}

  // Can't be fixed by parentheses.
  INIT(e, {1, 2, 3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-error@-2 {{use of undeclared identifier}}
  // expected-note@-3 {{cannot use initializer list at the beginning of a macro argument}}

  // Can't be fixed by parentheses.
  INIT(e, {1, 2, 3} + {1, 2, 3});
  // expected-error@-1 {{too many arguments provided}}
  // expected-error@-2 {{use of undeclared identifier}}
  // expected-note@-3 {{cannot use initializer list at the beginning of a macro argument}}
}

// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{145:11-145:11}:"("
// CHECK: fix-it:"{{.*}}macro_with_initializer_list.cpp":{145:23-145:23}:")"

#define M(name,a,b,c,d,e,f,g,h,i,j,k,l) \
  Foo name = a + b + c + d + e + f + g + h + i + j + k + l;
// expected-note@-2 2{{defined here}}
void test2() {
  M(F1, Foo(), Foo(), Foo(), Foo(), Foo(), Foo(),
        Foo(), Foo(), Foo(), Foo(), Foo(), Foo());

  M(F2, Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3},
        Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3}, Foo{1,2,3});
  // expected-error@-2 {{too many arguments provided}}
  // expected-note@-3 {{parentheses are required}}

  M(F3, {1,2,3}, {1,2,3}, {1,2,3}, {1,2,3}, {1,2,3}, {1,2,3},
        {1,2,3}, {1,2,3}, {1,2,3}, {1,2,3}, {1,2,3}, {1,2,3});
  // expected-error@-2 {{too many arguments provided}}
  // expected-error@-3 {{use of undeclared identifier}}
  // expected-note@-4 {{cannot use initializer list at the beginning of a macro argument}}
}
