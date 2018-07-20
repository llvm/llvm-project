// RUN: %clang_cc1 %s -ast-print -verify > %t.c
// RUN: FileCheck %s --input-file %t.c
//
// RUN: echo >> %t.c "// expected""-warning@* {{use of GNU old-style field designator extension}}"
// RUN: echo >> %t.c "// expected""-warning@* {{'EnumWithAttributes' is deprecated}}"
// RUN: echo >> %t.c "// expected""-note@* {{'EnumWithAttributes' has been explicitly marked deprecated here}}"
// RUN: echo >> %t.c "// expected""-warning@* {{'EnumWithAttributes2' is deprecated}}"
// RUN: echo >> %t.c "// expected""-note@* {{'EnumWithAttributes2' has been explicitly marked deprecated here}}"
// RUN: echo >> %t.c "// expected""-warning@* {{'EnumWithAttributes3' is deprecated}}"
// RUN: echo >> %t.c "// expected""-note@* {{'EnumWithAttributes3' has been explicitly marked deprecated here}}"
// RUN: %clang_cc1 -fsyntax-only %t.c -verify

typedef void func_typedef();
func_typedef xxx;

typedef void func_t(int x);
func_t a;

struct blah {
  struct {
    struct {
      int b;
    };
  };
};

// This used to crash clang.
struct {
}(s1);

int foo(const struct blah *b) {
  // CHECK: return b->b;
  return b->b;
}

int arr(int a[static 3]) {
  // CHECK: int a[static 3]
  return a[2];
}

int rarr(int a[restrict static 3]) {
  // CHECK: int a[restrict static 3]
  return a[2];
}

int varr(int n, int a[static n]) {
  // CHECK: int a[static n]
  return a[2];
}

int rvarr(int n, int a[restrict static n]) {
  // CHECK: int a[restrict static n]
  return a[2];
}

// CHECK: typedef struct {
typedef struct {
  int f;
} T __attribute__ ((__aligned__));

// CHECK: struct __attribute__((visibility("default"))) S;
struct __attribute__((visibility("default"))) S;

struct pair_t {
  int a;
  int b;
};

// CHECK: struct pair_t p = {a: 3, .b = 4};
struct pair_t p = {a: 3, .b = 4}; // expected-warning {{use of GNU old-style field designator extension}}

void initializers() {
  // CHECK: int *x = ((void *)0), *y = ((void *)0);
  int *x = ((void *)0), *y = ((void *)0);
  struct Z{};
  struct {
    struct Z z;
  // CHECK: } z = {(struct Z){}};
  } z = {(struct Z){}};
}

// CHECK-LABEL: enum __attribute__((deprecated(""))) EnumWithAttributes {
enum EnumWithAttributes { // expected-warning {{'EnumWithAttributes' is deprecated}}
  // CHECK-NEXT: EnumWithAttributesFoo __attribute__((deprecated(""))),
  EnumWithAttributesFoo __attribute__((deprecated)),
  // CHECK-NEXT: EnumWithAttributesBar __attribute__((unavailable(""))) = 50
  EnumWithAttributesBar __attribute__((unavailable)) = 50
  // CHECK-NEXT: } *EnumWithAttributesPtr;
} __attribute__((deprecated)) *EnumWithAttributesPtr; // expected-note {{'EnumWithAttributes' has been explicitly marked deprecated here}}

// CHECK-LABEL: enum __attribute__((deprecated(""))) EnumWithAttributes2 *EnumWithAttributes2Ptr;
// expected-warning@+2 {{'EnumWithAttributes2' is deprecated}}
// expected-note@+1 {{'EnumWithAttributes2' has been explicitly marked deprecated here}}
enum __attribute__((deprecated)) EnumWithAttributes2 *EnumWithAttributes2Ptr;

// CHECK-LABEL: EnumWithAttributes3Fn
void EnumWithAttributes3Fn() {
  // CHECK-NEXT: enum __attribute__((deprecated(""))) EnumWithAttributes3 *EnumWithAttributes3Ptr;
  // expected-warning@+2 {{'EnumWithAttributes3' is deprecated}}
  // expected-note@+1 {{'EnumWithAttributes3' has been explicitly marked deprecated here}}
  enum __attribute__((deprecated)) EnumWithAttributes3 *EnumWithAttributes3Ptr;
  // Printing must not put the attribute after the tag where it would apply to
  // the variable instead of the type, and then our deprecation warning would
  // move to this use of the variable.
  void *p = EnumWithAttributes3Ptr;
}
