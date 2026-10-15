// RUN: %clang_cc1 -x c -triple bpf-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

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
enum { A1 } __reloc__; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum without a typedef name}}
enum { A2 } __reloc__ v2; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum}}
typedef enum { A3 } __reloc__ *t3p; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum}}
struct s4 { enum { A4 } __reloc__ f; }; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum}}
enum __reloc__ { A9 } __reloc__; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum}}
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = enum)
enum { A10 } __reloc__; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum}}
#pragma clang attribute pop
int invalid3(void) {
  enum { A5 } __reloc__; // expected-warning {{'preserve_access_index' attribute has no effect on an anonymous enum}}
  return A1 + A2 + A3 + A4 + A5;
}

// No warning with a typedef name, an attribute implicitly added from the
// enclosing record, or #pragma clang attribute.
typedef enum __reloc__ { A6 } t6;
struct s7 { enum { A7 } f; } __reloc__;
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = enum)
enum { A8 };
#pragma clang attribute pop
int valid6(void) { return A6 + A7 + A8; }
