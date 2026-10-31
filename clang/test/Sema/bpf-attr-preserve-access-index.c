// RUN: %clang_cc1 -x c -triple bpf-pc-linux-gnu -debug-info-kind=limited -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c -triple bpf-pc-linux-gnu -fsyntax-only -verify %s

#define __reloc__ __attribute__((preserve_access_index))
#define __err_reloc__ __attribute__((preserve_access_index(0)))

struct t1 {
  int a;
  int b[4];
  int c:1;
} __reloc__;

union t2 {
  int a;
  int b[4];
  int c:1;
} __reloc__;

struct t3 {
  int a;
} __err_reloc__; // expected-error {{'preserve_access_index' attribute takes no arguments}}

struct t4 {
  union {
    int a;
    char b[5];
  };
  struct {
    int c:1;
  } __reloc__;
  int d;
} __reloc__;

struct __reloc__ p;
struct __reloc__ q;
struct p {
  int a;
};

int a __reloc__; // expected-error {{'preserve_access_index' attribute only applies to structs, unions, classes, and enums}}
struct s *p __reloc__; // expected-error {{'preserve_access_index' attribute only applies to structs, unions, classes, and enums}}

void invalid1(const int __reloc__ *arg) {} // expected-error {{'preserve_access_index' attribute only applies to structs, unions, classes, and enums}}
void invalid2(void) { const int __reloc__ *arg; } // expected-error {{'preserve_access_index' attribute only applies to structs, unions, classes, and enums}}
int valid3(struct t4 *arg) { return arg->a + arg->b[3] + arg->c + arg->d; }
int valid4(void *arg) {
  struct local_t { int a; int b; } __reloc__;
  return ((struct local_t *)arg)->b;
}

enum e1 { E1_A, E1_B } __reloc__;
enum __reloc__ e2 { E2_A };
typedef enum { E3_A } __reloc__ e3;
enum e4 { E4_A __reloc__ }; // expected-error {{'preserve_access_index' attribute only applies to structs, unions, classes, and enums}}
int valid5(void) { return E1_B + E2_A + E3_A; }

// An anonymous enum without typedef name cannot be relocated.
enum { A1 } __reloc__; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum without a typedef name}}
enum { A2 } __reloc__ v2; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum}}
typedef enum { A3 } __reloc__ *t3p; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum}}
struct s4 { enum { A4 } __reloc__ f; }; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum}}
enum __reloc__ { A9 } __reloc__; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum}}
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = enum)
enum { A10 } __reloc__; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum}}
#pragma clang attribute pop
int invalid3(void) {
  enum { A5 } __reloc__; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum}}
  return A1 + A2 + A3 + A4 + A5;
}

// No diagnostic with a typedef name, an attribute implicitly added from the
// enclosing record, or #pragma clang attribute.
typedef enum __reloc__ { A6 } t6;
struct s7 { enum { A7 } f; } __reloc__;
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = enum)
enum { A8 };
#pragma clang attribute pop
int valid6(void) { return A6 + A7 + A8; }

// An enumerator of an enum with the attribute is relocated, so it cannot be
// used where a compile-time constant is required.
int g1 = E1_B; // expected-error {{initializer element is not a compile-time constant}} expected-note {{enumerator 'E1_B' is not a constant expression since 'enum e1' has the 'preserve_access_index' attribute}}
int g2[E1_B]; // expected-error {{variable length array declaration not allowed at file scope}} expected-note {{enumerator 'E1_B' is not a constant expression since 'enum e1' has the 'preserve_access_index' attribute}}
struct s8 {
  int a[E1_B]; // expected-error {{fields must have a constant size}} expected-note {{enumerator 'E1_B' is not a constant expression since 'enum e1' has the 'preserve_access_index' attribute}}
  int b : E1_B; // expected-error {{expression is not an integer constant expression}} expected-note {{enumerator 'E1_B' is not a constant expression since 'enum e1' has the 'preserve_access_index' attribute}}
};
_Static_assert(E2_A == 0, ""); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{enumerator 'E2_A' is not a constant expression since 'enum e2' has the 'preserve_access_index' attribute}}
enum e7 { E7_A = A6 }; // expected-error {{expression is not an integer constant expression}} expected-note {{enumerator 'A6' is not a constant expression since 't6' has the 'preserve_access_index' attribute}}
int constant_contexts(int v) {
  static int s = E1_B; // expected-error {{initializer element is not a compile-time constant}} expected-note {{enumerator 'E1_B' is not a constant expression since 'enum e1' has the 'preserve_access_index' attribute}}
  switch (v) {
  case E1_B: // expected-error {{expression is not an integer constant expression}} expected-note {{enumerator 'E1_B' is not a constant expression since 'enum e1' has the 'preserve_access_index' attribute}}
    return 1;
  case 2 ... E2_A + 3: // expected-error {{expression is not an integer constant expression}} expected-note {{enumerator 'E2_A' is not a constant expression since 'enum e2' has the 'preserve_access_index' attribute}}
    return 2;
  }
  return s;
}

// An enumerator referenced in the definition of its enum, in an unevaluated
// operand, or of an anonymous enum without a typedef name, is not relocated.
// __builtin_preserve_enum_value() works as before.
enum e9 { E9_A = 1, E9_B = E9_A + 1 } __reloc__;
enum __reloc__ e10 { E10_A = 1, E10_B = E10_A + 1 };
_Static_assert(sizeof(E1_B) == sizeof(int), "");
int g4 = A1 + A7 + A8;
unsigned long valid7(int v) {
  switch (v) {
  case sizeof(E1_B):
    return 1;
  case A7 + 10:
    return 2;
  }
  return E1_B + __builtin_preserve_enum_value(*(enum e1 *)E1_A, 1);
}

// Only the argument of __builtin_preserve_enum_value() is checked as before.
// A relocated enumerator cast to a pointer is not warned on, as it was not
// before when it was a constant.
unsigned long casts(int v) {
  unsigned long r = __builtin_preserve_enum_value(*(int *)E1_A, 1); // expected-error {{__builtin_preserve_enum_value argument 1 invalid}}
  r += __builtin_preserve_enum_value(*(enum e1 *)(E1_A), 1); // expected-error {{__builtin_preserve_enum_value argument 1 invalid}}
  r += (unsigned long)(char *)E1_B + (unsigned long)(void *)(E1_B);
  return r + (unsigned long)(char *)v; // expected-warning {{cast to 'char *' from smaller integer type 'int'}}
}
