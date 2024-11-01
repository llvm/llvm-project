// RUN: %clang_cc1 -std=c23 -verify=ref,both -triple x86_64 -pedantic -Wno-conversion -Wno-constant-conversion -Wno-div-by-zero %s
// RUN: %clang_cc1 -std=c23 -verify=expected,both -triple x86_64 -pedantic -Wno-conversion -Wno-constant-conversion -Wno-div-by-zero -fexperimental-new-constant-interpreter %s

// Check that constexpr only applies to variables.
constexpr void f0() {} // both-error {{'constexpr' can only be used in variable declarations}}
constexpr const int f1() { return 0; } // both-error {{'constexpr' can only be used in variable declarations}}

constexpr struct S1 { int f; }; //both-error {{struct cannot be marked constexpr}}
constexpr struct S2 ; // both-error {{struct cannot be marked constexpr}}
constexpr union U1; // both-error {{union cannot be marked constexpr}}
constexpr union U2 {int a; float b;}; // both-error {{union cannot be marked constexpr}}
constexpr enum E1 {A = 1, B = 2} ; // both-error {{enum cannot be marked constexpr}}
struct S3 {
  static constexpr int f = 0; // both-error {{type name does not allow storage class}}
  // both-error@-1 {{type name does not allow constexpr}}
  // both-error@-2 {{expected ';' at end}}
  constexpr int f1 = 0;
  // both-error@-1 {{type name does not allow constexpr}}
  // both-error@-2 {{expected ';' at end}}
};

constexpr; // both-error {{'constexpr' can only be used in variable declarations}}
constexpr int V1 = 3;
constexpr float V2 = 7.0;
int V3 = (constexpr)3; // both-error {{expected expression}}

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
constexpr auto Ulong = 1L;
constexpr auto CompoundLiteral = (int){13};
constexpr auto DoubleCast = (double)(1 / 3);
constexpr auto String = "this is a string"; // both-error {{constexpr pointer initializer is not null}}
constexpr signed auto Long = 1L; // both-error {{'auto' cannot be signed or unsigned}}
_Static_assert(_Generic(Ulong, long : 1));
_Static_assert(_Generic(CompoundLiteral, int : 1));
_Static_assert(_Generic(DoubleCast, double : 1));
_Static_assert(_Generic(String, char* : 1));

typedef constexpr int Foo; // both-error {{typedef cannot be constexpr}}
constexpr typedef int Bar; // both-error {{typedef cannot be constexpr}}

void f3(constexpr register int P1) { // both-error {{function parameter cannot be constexpr}}
  constexpr register int V9 = 0;
  constexpr register auto V10 = 0.0;
}

constexpr thread_local int V11 = 38; // both-error {{cannot combine with previous '_Thread_local' declaration specifier}}
constexpr static thread_local double V12 = 38; // both-error {{cannot combine with previous '_Thread_local' declaration specifier}}
constexpr extern thread_local char V13; // both-error {{cannot combine with previous '_Thread_local' declaration specifier}}
// both-error@-1 {{cannot combine with previous 'extern' declaration specifier}}
// both-error@-2 {{constexpr variable declaration must be a definition}}
constexpr thread_local short V14 = 38; // both-error {{cannot combine with previous '_Thread_local' declaration specifier}}

// Check how constexpr works with qualifiers.
constexpr _Atomic int V15 = 0; // both-error {{constexpr variable cannot have type 'const _Atomic(int)'}}
constexpr _Atomic(int) V16 = 0; // both-error {{constexpr variable cannot have type 'const _Atomic(int)'}}

constexpr volatile int V17 = 0; // both-error {{constexpr variable cannot have type 'const volatile int'}}

constexpr int * restrict V18 = 0; // both-error {{constexpr variable cannot have type 'int *const restrict'}}

constexpr extern char Oops = 1; // both-error {{cannot combine with previous 'extern' declaration specifier}} \
                                // both-warning {{'extern' variable has an initializer}}

constexpr int * restrict * Oops1 = 0;

typedef _Atomic(int) TheA;
typedef volatile short TheV;
typedef float * restrict TheR;

constexpr TheA V19[3] = {};
// both-error@-1 {{constexpr variable cannot have type 'const TheA[3]' (aka 'const _Atomic(int)[3]')}}
constexpr TheV V20[3] = {};
// both-error@-1 {{constexpr variable cannot have type 'const TheV[3]' (aka 'const volatile short[3]')}}
constexpr TheR V21[3] = {};
// both-error@-1 {{constexpr variable cannot have type 'const TheR[3]' (aka 'float *restrict const[3]')}}

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
// both-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}
constexpr struct HasV V23[2] = {};
// both-error@-1 {{constexpr variable cannot have type 'TheV' (aka 'volatile short')}}
constexpr struct HasR V24[2] = {};
// both-error@-1 {{constexpr variable cannot have type 'TheR' (aka 'float *restrict')}}

union U3 {
  float a;
  union {
    struct HasA f;
    struct HasR f1;
  };
};

constexpr union U3 V25 = {};
// both-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}
constexpr union U3 V26[8] = {};
// both-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}

struct S4 {
  union U3 f[3];
};

constexpr struct S4 V27 = {};
// both-error@-1 {{constexpr variable cannot have type 'TheA' (aka '_Atomic(int)')}}
constexpr const int V28 = 28;

struct S {
  union {
    volatile int i;
  };
  int j;
};

constexpr struct S s = {}; // both-error {{constexpr variable cannot have type 'volatile int'}}

// Check that constexpr variable must have a valid initializer which is a
// constant expression.
constexpr int V29;
// both-error@-1 {{constexpr variable 'V29' must be initialized by a constant expression}}

struct S5 {
  int f;
};

constexpr struct S5 V30;
// both-error@-1 {{constexpr variable 'V30' must be initialized by a constant expression}}
constexpr struct S5 V31 = {};

int randomFoo() { return 7; }

constexpr float V32 = randomFoo();
// both-error@-1 {{constexpr variable 'V32' must be initialized by a constant expression}}

const int V33 = 4;
const int V34 = 0;
const int V35 = 2;

constexpr int V36 = V33 / V34;
// both-error@-1 {{constexpr variable 'V36' must be initialized by a constant expression}}
constexpr int V37 = V33 / V35;
// both-error@-1 {{constexpr variable 'V37' must be initialized by a constant expression}}
constexpr int V38 = 3;
constexpr int V39 = V38 / V38;
constexpr int V40 = V38 / 2;
constexpr int V41 = V38 / 0;
// both-error@-1 {{constexpr variable 'V41' must be initialized by a constant expression}}
// both-note@-2 {{division by zero}}
constexpr int V42 = V38 & 0;

constexpr struct S5 V43 = { randomFoo() };
// both-error@-1 {{constexpr variable 'V43' must be initialized by a constant expression}}
constexpr struct S5 V44 = { 0 };
constexpr struct S5 V45 = { V38 / 0 };
// both-error@-1 {{constexpr variable 'V45' must be initialized by a constant expression}}
// both-note@-2 {{division by zero}}

constexpr float V46[3] = {randomFoo() };
// both-error@-1 {{constexpr variable 'V46' must be initialized by a constant expression}}
constexpr struct S5 V47[3] = {randomFoo() };
// both-error@-1 {{constexpr variable 'V47' must be initialized by a constant expression}}

const static int V48 = V38;
constexpr static int V49 = V48;
// both-error@-1 {{constexpr variable 'V49' must be initialized by a constant expression}}

void f4(const int P1) {
  constexpr int V = P1;
// both-error@-1 {{constexpr variable 'V' must be initialized by a constant expression}}

  constexpr int V1 = 12;
  constexpr const int *V2 = &V1;
// both-error@-1 {{constexpr variable 'V2' must be initialized by a constant expression}}
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

struct S8 {
  unsigned char a[3];
  unsigned int b[3];
};

constexpr struct S8 DesigInit = {.b = {299, 7, 8}, .a = {-1, 7, 8}};
// both-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'unsigned char'}}

void f5() {
  constexpr char V50 = 300;
  // both-error@-1 {{constexpr initializer evaluates to 300 which is not exactly representable in type 'const char'}}
  constexpr float V51 = 1.0 / 3.0;
  // both-error@-1 {{constexpr initializer evaluates to 3.333333e-01 which is not exactly representable in type 'const float'}}
  constexpr float V52 = 0.7;
  // both-error@-1 {{constexpr initializer evaluates to 7.000000e-01 which is not exactly representable in type 'const float'}}
  constexpr float V53 = 1.0f / 3.0f;
  constexpr float V54 = 432000000000;
  // both-error@-1 {{constexpr initializer evaluates to 432000000000 which is not exactly representable in type 'const float'}}
  constexpr unsigned char V55[] = {
      "\xAF",
  // both-error@-1 {{constexpr initializer evaluates to -81 which is not exactly representable in type 'const unsigned char'}}
  };

  constexpr unsigned char V56[] = {
      u8"\xAF",
  };
  constexpr struct S6 V57 = {299};
  // both-error@-1 {{constexpr initializer evaluates to 299 which is not exactly representable in type 'unsigned char'}}
  constexpr struct S6 V58 = {-299};
  // both-error@-1 {{constexpr initializer evaluates to -299 which is not exactly representable in type 'unsigned char'}}
  constexpr double V59 = 0.5;
  constexpr double V60 = 1.0;
  constexpr float V61 = V59 / V60;
  constexpr double V62 = 1.7;
  constexpr float V63 = V59 / V62;
  // both-error@-1 {{constexpr initializer evaluates to 2.941176e-01 which is not exactly representable in type 'const float'}}

  constexpr unsigned char V64 = '\xAF';
  // both-error@-1 {{constexpr initializer evaluates to -81 which is not exactly representable in type 'const unsigned char'}}
  constexpr unsigned char V65 = u8'\xAF';

  constexpr char V66[3] = {300};
  // both-error@-1 {{constexpr initializer evaluates to 300 which is not exactly representable in type 'const char'}}
  constexpr struct S6 V67[3] = {300};
  // both-error@-1 {{constexpr initializer evaluates to 300 which is not exactly representable in type 'unsigned char'}}

  constexpr struct S7 V68 = {0.3, -1 };
  // both-error@-1 {{constexpr initializer evaluates to 3.000000e-01 which is not exactly representable in type 'float'}}
  // both-error@-2 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'unsigned int'}}
  constexpr struct S7 V69 = {0.5, -1 };
  // both-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'unsigned int'}}
  constexpr struct S7 V70[3] = {{123456789}};
  // both-error@-1 {{constexpr initializer evaluates to 123456789 which is not exactly representable in type 'float'}}

  constexpr int V71 = 0.3;
  // both-error@-1 {{constexpr initializer for type 'const int' is of type 'double'}}
  constexpr int V72 = V59;
  // both-error@-1 {{constexpr initializer for type 'const int' is of type 'const double'}}
  constexpr struct S6 V73 = {V59};
  // both-error@-1 {{constexpr initializer for type 'unsigned char' is of type 'const double'}}

  constexpr float V74 = 1;
  constexpr float V75 = V59;
  constexpr unsigned int V76[3] = {0.5};
  // both-error@-1 {{constexpr initializer for type 'const unsigned int' is of type 'double'}}

  constexpr _Complex float V77 = 0;
  constexpr float V78 = V77;
  // both-error@-1 {{constexpr initializer for type 'const float' is of type 'const _Complex float'}}
  constexpr int V79 = V77;
  // both-error@-1 {{constexpr initializer for type 'const int' is of type 'const _Complex float'}}

}

constexpr char string[] = "test""ing this out\xFF";
constexpr unsigned char ustring[] = "test""ing this out\xFF";
// both-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned char'}}
constexpr char u8string[] = u8"test"u8"ing this out\xFF";
// both-error@-1 {{constexpr initializer evaluates to 255 which is not exactly representable in type 'const char'}}
constexpr unsigned char u8ustring[] = u8"test"u8"ing this out\xFF";
constexpr unsigned short uustring[] = u"test"u"ing this out\xFF";
constexpr unsigned int Ustring[] = U"test"U"ing this out\xFF";
constexpr unsigned char Arr2[6][6] = {
  {"ek\xFF"}, {"ek\xFF"}
// both-error@-1 2{{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned char'}}
};

constexpr int i = (12);
constexpr int j = (i);
constexpr unsigned jneg = (-i);
// both-error@-1 {{constexpr initializer evaluates to -12 which is not exactly representable in type 'const unsigned int'}}

// Check that initializer for pointer constexpr variable should be null.
constexpr int V80 = 3;
constexpr const int *V81 = &V80;
// both-error@-1 {{constexpr pointer initializer is not null}}
constexpr int *V82 = 0;
constexpr int *V83 = V82;
constexpr int *V84 = 42;
// ref-error@-1 {{constexpr variable 'V84' must be initialized by a constant expression}}
// ref-note@-2 {{this conversion is not allowed in a constant expression}}
// both-error@-3 {{constexpr pointer initializer is not null}}
constexpr int *V85 = nullptr;

// Check that constexpr variables should not be VLAs.
void f6(const int P1) {
  constexpr int V86[P1] = {};
// both-error@-1 {{constexpr variable cannot have type 'const int[P1]'}}
  const int V87 = 3;
  constexpr int V88[V87] = {};
// both-warning@-1 {{variable length array folded to constant array as an extension}}
  int V89 = 7;
  constexpr int V90[V89] = {};
// both-error@-1 {{constexpr variable cannot have type 'const int[V89]'}}
}

void f7(int n, int array[n]) {
  constexpr typeof(array) foo = 0; // Accepted because array is a pointer type, not a VLA type
  int (*(*fp)(int n))[n];
  constexpr typeof(fp) bar = 0; // both-error {{constexpr variable cannot have type 'const typeof (fp)' (aka 'int (*(*const)(int))[n]')}}
}

// Check how constexpr works with NaNs and infinities.
#define FLT_NAN __builtin_nanf("1")
#define DBL_NAN __builtin_nan("1")
#define LD_NAN __builtin_nanf("1")
#define FLT_SNAN __builtin_nansf("1")
#define DBL_SNAN __builtin_nans("1")
#define LD_SNAN __builtin_nansl("1")
#define INF __builtin_inf()
void infsNaNs() {
  // Inf and quiet NaN is always fine, signaling NaN must have the same type.
  constexpr float fl0 = INF;
  constexpr float fl1 = (long double)INF;
  constexpr float fl2 = (long double)FLT_NAN;
  constexpr float fl3 = FLT_NAN;
  constexpr float fl5 = DBL_NAN;
  constexpr float fl6 = LD_NAN;
  constexpr float fl7 = DBL_SNAN; // both-error {{constexpr initializer evaluates to nan which is not exactly representable in type 'const float'}}
  constexpr float fl8 = LD_SNAN; // both-error {{constexpr initializer evaluates to nan which is not exactly representable in type 'const float'}}

  constexpr double db0 = FLT_NAN;
  constexpr double db2 = DBL_NAN;
  constexpr double db3 = DBL_SNAN;
  constexpr double db4 = FLT_SNAN; // both-error {{constexpr initializer evaluates to nan which is not exactly representable in type 'const double'}}
  constexpr double db5 = LD_SNAN; // both-error {{constexpr initializer evaluates to nan which is not exactly representable in type 'const double'}}
  constexpr double db6 = INF;
}

constexpr struct S9 s9 = {  }; // both-error {{variable has incomplete type 'const struct S9'}} \
                               // both-note {{forward declaration of 'struct S9'}}

struct S10 {
  signed long long i : 8;
};
constexpr struct S10 c = { 255 };
// FIXME-expected-error@-1 {{constexpr initializer evaluates to 255 which is not exactly representable in 'long long' bit-field with width 8}}
// See: GH#101299
