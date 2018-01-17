// RUN: %clang_cc1 %s -ast-print | FileCheck %s
// RUN: %clang_cc1 %s -ast-print | %clang_cc1 -fsyntax-only -

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
struct pair_t p = {a: 3, .b = 4};

void initializers() {
  // CHECK: int *x = ((void *)0), *y = ((void *)0);
  int *x = ((void *)0), *y = ((void *)0);
  struct Z{};
  struct {
    struct Z z;
  // CHECK: } z = {(struct Z){}};
  } z = {(struct Z){}};
}

// CHECK-LABEL: enum EnumWithAttributes {
enum EnumWithAttributes {
  // CHECK-NEXT: EnumWithAttributesFoo __attribute__((deprecated(""))),
  EnumWithAttributesFoo __attribute__((deprecated)),
  // CHECK-NEXT: EnumWithAttributesBar __attribute__((unavailable(""))) = 50
  EnumWithAttributesBar __attribute__((unavailable)) = 50
  // CHECK-NEXT: } __attribute__((deprecated("")))
} __attribute__((deprecated));
