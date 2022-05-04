// RUN: %clang_cc1 -fsyntax-only -fdouble-square-bracket-attributes -verify=expected,notc2x -Wno-strict-prototypes %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu2x -verify=expected,c2x %s

enum [[]] E {
  One [[]],
  Two,
  Three [[]]
};

enum [[]] { Four };
[[]] enum E2 { Five }; // expected-error {{misplaced attributes}}

// FIXME: this diagnostic can be improved.
enum { [[]] Six }; // expected-error {{expected identifier}}

// FIXME: this diagnostic can be improved.
enum E3 [[]] { Seven }; // expected-error {{expected identifier or '('}}

[[deprecated([""])]] int WrongArgs; // expected-error {{expected expression}}
[[,,,,,]] int Commas1; // ok
[[,, maybe_unused]] int Commas2; // ok
[[maybe_unused,,,]] int Commas3; // ok
[[,,maybe_unused,]] int Commas4; // ok
[[foo bar]] int NoComma; // expected-error {{expected ','}} \
                         // expected-warning {{unknown attribute 'foo' ignored}}

struct [[]] S1 {
  int i [[]];
  int [[]] j;
  int k[10] [[]];
  int l[[]][10];
  [[]] int m, n;
  int o [[]] : 12;
  int [[]] : 0; // OK, attribute applies to the type.
  int p, [[]] : 0; // expected-error {{an attribute list cannot appear here}}
  int q, [[]] r; // expected-error {{an attribute list cannot appear here}}
};

[[]] struct S2 { int a; }; // expected-error {{misplaced attributes}}
struct S3 [[]] { int a; }; // expected-error {{an attribute list cannot appear here}}

union [[]] U {
  double d [[]];
  [[]] int i;
};

[[]] union U2 { double d; }; // expected-error {{misplaced attributes}}
union U3 [[]] { double d; }; // expected-error {{an attribute list cannot appear here}}

struct [[]] IncompleteStruct;
union [[]] IncompleteUnion;
enum [[]] IncompleteEnum;
enum __attribute__((deprecated)) IncompleteEnum2;

[[]] void f1(void);
void [[]] f2(void);
void f3 [[]] (void);
void f4(void) [[]];

void f5(int i [[]], [[]] int j, int [[]] k);

void f6(a, b) [[]] int a; int b; { // notc2x-error {{an attribute list cannot appear here}} \
                                      c2x-error {{unknown type name 'a'}} \
                                      c2x-error {{unknown type name 'b'}} \
                                      c2x-error {{expected ';' after top level declarator}} \
                                      c2x-error {{expected identifier or '('}}
}

// FIXME: technically, an attribute list cannot appear here, but we currently
// parse it as part of the return type of the function, which is reasonable
// behavior given that we *don't* want to parse it as part of the K&R parameter
// declarations. It is disallowed to avoid a parsing ambiguity we already
// handle well.
int (*f7(a, b))(int, int) [[]] int a; int b; { // c2x-error {{unknown type name 'a'}} \
                                                  c2x-error {{unknown type name 'b'}} \
                                                  c2x-error {{expected ';' after top level declarator}} \
                                                  c2x-error {{expected identifier or '('}}

  return 0;
}

[[]] int a, b;
int c [[]], d [[]];

void f8(void) [[]] {
  [[]] int i, j;
  int k, l [[]];
}

[[]] void f9(void) {
  int i[10] [[]];
  int (*fp1)(void)[[]];
  int (*fp2 [[]])(void);

  int * [[]] *ipp;
}

void f10(int j[static 10] [[]], int k[*] [[]]);

void f11(void) {
  [[]] {}
  [[]] if (1) {}

  [[]] switch (1) {
  [[]] case 1: [[]] break;
  [[]] default: break;
  }

  goto foo;
  [[]] foo: (void)1;

  [[]] for (;;);
  [[]] while (1);
  [[]] do [[]] { } while(1);

  [[]] (void)1;

  [[]];

  (void)sizeof(int [4][[]]);
  (void)sizeof(struct [[]] S3 { int a [[]]; });

  [[]] return;
}

[[attr]] void f12(void); // expected-warning {{unknown attribute 'attr' ignored}}
[[vendor::attr]] void f13(void); // expected-warning {{unknown attribute 'attr' ignored}}

// Ensure that asm statements properly handle double colons.
void test_asm(void) {
  asm("ret" :::);
  asm("foo" :: "r" (xx)); // expected-error {{use of undeclared identifier 'xx'}}
}

// Do not allow 'using' to introduce vendor attribute namespaces.
[[using vendor: attr1, attr2]] void f14(void); // expected-error {{expected ','}} \
                                               // expected-warning {{unknown attribute 'using' ignored}}

struct [[]] S4 *s; // expected-error {{an attribute list cannot appear here}}
struct S5 {};
int c = sizeof(struct [[]] S5); // expected-error {{an attribute list cannot appear here}}
