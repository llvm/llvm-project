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
typedef enum { A3 } __reloc__ *t3p; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum without a typedef name}}
enum __reloc__ { A9 } __reloc__; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum without a typedef name}}
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = enum)
enum { A10 } __reloc__; // expected-error {{'preserve_access_index' attribute has no effect on an anonymous enum without a typedef name}}
#pragma clang attribute pop

// No diagnostic with a typedef name, in a struct with the attribute, or with
// #pragma clang attribute.
typedef enum __reloc__ { A6 } t6;
struct s7 { enum { A7 } f; } __reloc__;
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = enum)
enum { A8 };
#pragma clang attribute pop
int valid6(void) { return A6 + A7 + A8; }

// With #pragma clang attribute, an enum defined in a struct gets the
// attribute.
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = any(record, enum))
struct s10 { enum e12 { E12_A } f; };
#pragma clang attribute pop
_Static_assert(E12_A == 0, ""); // expected-error {{enumerator 'E12_A' is not relocated where a constant is required, although 'enum e12' has the 'preserve_access_index' attribute}}

// An array bound in a function, or in the operand of sizeof or typeof, is
// kept a constant. Neither becomes variably modified.
int local_array(int i) {
  int a[E1_B]; // expected-error {{enumerator 'E1_B' is not relocated where a constant is required, although 'enum e1' has the 'preserve_access_index' attribute}}
  a[i] = 1;
  return a[0];
}
unsigned long vm_operand(void *p) {
  typeof(*(int (*)[E1_B])p) *q = p; // expected-error {{enumerator 'E1_B' is not relocated where a constant is required, although 'enum e1' has the 'preserve_access_index' attribute}}
  return sizeof(*(int (*)[E1_B])p) + sizeof(*q); // expected-error {{enumerator 'E1_B' is not relocated where a constant is required, although 'enum e1' has the 'preserve_access_index' attribute}}
}
enum e7 { E7_A = A6 }; // expected-error {{enumerator 'A6' is not relocated where a constant is required, although 't6' has the 'preserve_access_index' attribute}}

// An enumerator referenced in the definition of its enum, in an unevaluated
// operand, or of an anonymous enum without a typedef name, is not relocated.
enum __reloc__ e10 { E10_A = 1, E10_B = E10_A + 1 };
_Static_assert(sizeof(E1_B) == sizeof(int), "");
int g4 = A1 + A7 + A8;

// Only the argument of __builtin_preserve_enum_value() is checked as before.
unsigned long casts(void) {
  unsigned long r = __builtin_preserve_enum_value(*(int *)E1_A, 1); // expected-error {{__builtin_preserve_enum_value argument 1 invalid}}
  r += __builtin_preserve_enum_value(*(enum e1 *)(E1_A), 1); // expected-error {{__builtin_preserve_enum_value argument 1 invalid}}
  return r;
}

// An inline asm operand that can only be an immediate is kept a constant only
// if a relocated enumerator makes it non-constant.
void asm_imm(void) {
  asm volatile("r0 = %0" :: "n"(E1_B + 1) : "r0"); // expected-error {{enumerator 'E1_B' is not relocated where a constant is required, although 'enum e1' has the 'preserve_access_index' attribute}}
  asm volatile("r0 = %0" :: "ri"(E1_B) : "r0");
  asm volatile("r0 = %0" :: "i"(0 && E1_B) : "r0");
  asm volatile("r0 = %0" :: "i"(sizeof(E1_B)) : "r0");
}
