// RUN: %clang_cc1 -std=c2x -verify -triple x86_64 -pedantic -Wno-conversion -Wno-constant-conversion -Wno-div-by-zero %s

// Check that constexpr only applies to variables.
constexpr void f0() {} // expected-error {{'constexpr' can only be used in variable declarations}}
constexpr const int f1() { return 0; } // expected-error {{'constexpr' can only be used in variable declarations}}

constexpr struct S1 { int f; }; //expected-error {{struct cannot be marked constexpr}}
constexpr struct S2 ; // expected-error {{struct cannot be marked constexpr}}
constexpr union U1; // expected-error {{union cannot be marked constexpr}}
constexpr union U2 {int a; float b;}; // expected-error {{union cannot be marked constexpr}}
constexpr enum E1 {A = 1, B = 2} ; // expected-error {{enum cannot be marked constexpr}}
struct S3 {
  static constexpr int f = 0; // expected-error {{type name does not allow storage class}}
  // expected-error@-1 {{type name does not allow constexpr}}
  // expected-error@-2 {{expected ';' at end}}
  constexpr int f1 = 0;
  // expected-error@-1 {{type name does not allow constexpr}}
  // expected-error@-2 {{expected ';' at end}}
};

constexpr; // expected-error {{'constexpr' can only be used in variable declarations}}
constexpr int V1 = 3;
constexpr float V2 = 7.0;
int V3 = (constexpr)3; // expected-error {{expected expression}}

void f2() {
  constexpr int a = 0;
  constexpr float b = 1.7f;
}

// Check how constexpr works with other storage-class specifiers.
constexpr auto V4 = 1;
constexpr static auto V5 = 1;
constexpr static const auto V6 = 1;
constexpr static const int V7 = 1;
constexpr static int V8 = 1;

void f3(constexpr register int P1) { // expected-error {{function parameter cannot be constexpr}}
  constexpr register int V9 = 0;
  constexpr register auto V10 = 0.0;
}

constexpr thread_local int V11 = 38; // expected-error {{cannot combine with previous '_Thread_local' declaration specifier}}
constexpr static thread_local double V12 = 38; // expected-error {{cannot combine with previous '_Thread_local' declaration specifier}}
constexpr extern thread_local char V13; // expected-error {{cannot combine with previous '_Thread_local' declaration specifier}}
// expected-error@-1 {{cannot combine with previous 'extern' declaration specifier}}
// expected-error@-2 {{constexpr variable declaration must be a definition}}
constexpr thread_local short V14 = 38; // expected-error {{cannot combine with previous '_Thread_local' declaration specifier}}

// Check how constexpr works with qualifiers.
constexpr _Atomic int V15 = 0; // expected-error {{constexpr variable cannot have type 'const _Atomic(int)'}}
constexpr _Atomic(int) V16 = 0; // expected-error {{constexpr variable cannot have type 'const _Atomic(int)'}}

constexpr volatile int V17 = 0; // expected-error {{constexpr variable cannot have type 'const volatile int'}}

constexpr int * restrict V18 = 0; // expected-error {{constexpr variable cannot have type 'int *const restrict'}}

typedef _Atomic(int) TheA;
typedef volatile short TheV;
typedef float * restrict TheR;

constexpr TheA V19[3] = {};
// expected-error@-1 {{constexpr variable cannot have type 'const TheA' (aka 'const _Atomic(int)')}}
constexpr TheV V20[3] = {};
// expected-error@-1 {{constexpr variable cannot have type 'const TheV' (aka 'const volatile short')}}
constexpr TheR V21[3] = {};
// expected-error@-1 {{constexpr variable cannot have type 'const TheR' (aka 'float *const restrict')}}

struct HasA {
  TheA f;
  int b;
};

struct HasV {
  float b;
  TheV f;
};

struct HasR {
  short b;
  int a;
  TheR f;
};

constexpr struct HasA V22[2] = {};
// expected-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}
constexpr struct HasV V23[2] = {};
// expected-error@-1 {{constexpr variable cannot have type 'TheV' (aka 'volatile short')}}
constexpr struct HasR V24[2] = {};
// expected-error@-1 {{constexpr variable cannot have type 'TheR' (aka 'float *restrict')}}

union U3 {
  float a;
  union {
    struct HasA f;
    struct HasR f1;
  };
};

constexpr union U3 V25 = {};
// expected-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}
constexpr union U3 V26[8] = {};
// expected-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}

struct S4 {
  union U3 f[3];
};

constexpr struct S4 V27 = {};
// expected-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}
constexpr const int V28 = 28;

// Check that constexpr variable must have a valid initializer which is a
// constant expression.
constexpr int V29;
// expected-error@-1 {{constexpr variable 'V29' must be initialized by a constant expression}}

struct S5 {
  int f;
};

constexpr struct S5 V30;
// expected-error@-1 {{constexpr variable 'V30' must be initialized by a constant expression}}
constexpr struct S5 V31 = {};

int randomFoo() { return 7; }

constexpr float V32 = randomFoo();
// expected-error@-1 {{constexpr variable 'V32' must be initialized by a constant expression}}

const int V33 = 4;
const int V34 = 0;
const int V35 = 2;

constexpr int V36 = V33 / V34;
// expected-error@-1 {{constexpr variable 'V36' must be initialized by a constant expression}}
constexpr int V37 = V33 / V35;
// expected-error@-1 {{constexpr variable 'V37' must be initialized by a constant expression}}
constexpr int V38 = 3;
constexpr int V39 = V38 / V38;
constexpr int V40 = V38 / 2;
constexpr int V41 = V38 / 0;
// expected-error@-1 {{constexpr variable 'V41' must be initialized by a constant expression}}
// expected-note@-2 {{division by zero}}
constexpr int V42 = V38 & 0;

constexpr struct S5 V43 = { randomFoo() };
// expected-error@-1 {{constexpr variable 'V43' must be initialized by a constant expression}}
constexpr struct S5 V44 = { 0 };
constexpr struct S5 V45 = { V38 / 0 };
// expected-error@-1 {{constexpr variable 'V45' must be initialized by a constant expression}}
// expected-note@-2 {{division by zero}}

constexpr float V46[3] = {randomFoo() };
// expected-error@-1 {{constexpr variable 'V46' must be initialized by a constant expression}}
constexpr struct S5 V47[3] = {randomFoo() };
// expected-error@-1 {{constexpr variable 'V47' must be initialized by a constant expression}}

const static int V48 = V38;
constexpr static int V49 = V48;
// expected-error@-1 {{constexpr variable 'V49' must be initialized by a constant expression}}

void f4(const int P1) {
  constexpr int V = P1;
// expected-error@-1 {{constexpr variable 'V' must be initialized by a constant expression}}

  constexpr int V1 = 12;
  constexpr const int *V2 = &V1;
// expected-error@-1 {{constexpr variable 'V2' must be initialized by a constant expression}}
// expected-error@-2 {{constexpr pointer initializer is not null}}
}

// Check that initializer for constexpr variable should match the type of the
// variable and is exactly representable int the variable's type.

struct S6 {
  unsigned char a;
};

struct S7 {
  union {
    float a;
  };
  unsigned int b;
};

void f5() {
  constexpr char V50 = 300;
  // expected-error@-1 {{constexpr initializer evaluates to 300 which is not exactly representable in type 'char'}}
  constexpr float V51 = 1.0 / 3.0;
  // expected-error@-1 {{constexpr initializer evaluates to 3.333333e-01 which is not exactly representable in type 'float'}}
  constexpr float V52 = 0.7;
  // expected-error@-1 {{constexpr initializer evaluates to 7.000000e-01 which is not exactly representable in type 'float'}}
  constexpr float V53 = 1.0f / 3.0f;
  constexpr float V54 = 432000000000;
  // expected-error@-1 {{constexpr initializer evaluates to 432000000000 which is not exactly representable in type 'float'}}
  constexpr unsigned char V55[] = {
      "\xAF",
  // expected-error@-1 {{constexpr initializer evaluates to -81 which is not exactly representable in type 'const unsigned char'}}
  };

  // FIXME Shouldn't be diagnosed if char_8t is supported.
  constexpr unsigned char V56[] = {
      u8"\xAF",
  // expected-error@-1 {{constexpr initializer evaluates to -81 which is not exactly representable in type 'const unsigned char'}}
  };
  constexpr struct S6 V57 = {299};
  // expected-error@-1 {{constexpr initializer evaluates to 299 which is not exactly representable in type 'unsigned char'}}
  constexpr struct S6 V58 = {-299};
  // expected-error@-1 {{constexpr initializer evaluates to -299 which is not exactly representable in type 'unsigned char'}}
  constexpr double V59 = 0.5;
  constexpr double V60 = 1.0;
  constexpr float V61 = V59 / V60;
  constexpr double V62 = 1.7;
  constexpr float V63 = V59 / V62;
  // expected-error@-1 {{constexpr initializer evaluates to 2.941176e-01 which is not exactly representable in type 'float'}}

  constexpr unsigned char V64 = '\xAF';
  // expected-error@-1 {{constexpr initializer evaluates to -81 which is not exactly representable in type 'unsigned char'}}
  constexpr unsigned char V65 = u8'\xAF';

  constexpr char V66[3] = {300};
  // expected-error@-1 {{constexpr initializer evaluates to 300 which is not exactly representable in type 'char'}}
  constexpr struct S6 V67[3] = {300};
  // expected-error@-1 {{constexpr initializer evaluates to 300 which is not exactly representable in type 'unsigned char'}}

  constexpr struct S7 V68 = {0.3, -1 };
  // expected-error@-1 {{constexpr initializer evaluates to 3.000000e-01 which is not exactly representable in type 'float'}}
  constexpr struct S7 V69 = {0.5, -1 };
  // expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'unsigned int'}}
  constexpr struct S7 V70[3] = {{123456789}};
  // expected-error@-1 {{constexpr initializer evaluates to 123456789 which is not exactly representable in type 'float'}}

  constexpr int V71 = 0.3;
  // expected-error@-1 {{constexpr initializer for type 'int' is of type 'double'}}
  constexpr int V72 = V59;
  // expected-error@-1 {{constexpr initializer for type 'int' is of type 'const double'}}
  constexpr struct S6 V73 = {V59};
  // expected-error@-1 {{constexpr initializer for type 'unsigned char' is of type 'const double'}}

  constexpr float V74 = 1;
  constexpr float V75 = V59;
  constexpr unsigned int V76[3] = {0.5};
  // expected-error@-1 {{constexpr initializer for type 'unsigned int' is of type 'double'}}

  constexpr _Complex float V77 = 0;
  constexpr float V78 = V77;
  // expected-error@-1 {{constexpr initializer for type 'float' is of type 'const _Complex float'}}
  constexpr int V79 = V77;
  // expected-error@-1 {{constexpr initializer for type 'int' is of type 'const _Complex float'}}

}

constexpr char string[] = "test""ing this out\xFF";
constexpr unsigned char ustring[] = "test""ing this out\xFF";
// expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned char'}}
constexpr char wstring[] = u8"test"u8"ing this out\xFF";
constexpr unsigned char wustring[] = u8"test"u8"ing this out\xFF";
// expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned char'}}

constexpr int i = (12);
constexpr int j = (i);

// Check that initializer for pointer constexpr variable should be null.
constexpr int V80 = 3;
constexpr const int *V81 = &V80;
// expected-error@-1 {{constexpr pointer initializer is not null}}
constexpr int *V82 = 0;
constexpr int *V83 = V82;
constexpr int *V84 = 42;
// expected-error@-1 {{constexpr variable 'V84' must be initialized by a constant expression}}
// expected-note@-2 {{this conversion is not allowed in a constant expression}}
constexpr int *V85 = nullptr;

// Check that constexpr variables should not be VLAs.
void f6(const int P1) {
  constexpr int V86[P1] = {};
// expected-error@-1 {{constexpr variable cannot have type 'const int[P1]'}}
  const int V87 = 3;
  constexpr int V88[V87] = {};
// expected-warning@-1 {{variable length array folded to constant array as an extension}}
  int V89 = 7;
  constexpr int V90[V89] = {};
// expected-error@-1 {{constexpr variable cannot have type 'const int[V89]'}}
}
