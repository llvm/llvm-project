// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.core.FixedAddr,alpha.core.PointerArithm,alpha.core.PointerSub,debug.ExprInspection -analyzer-store=region -verify -triple x86_64-apple-darwin9 -Wno-tautological-pointer-compare %s
// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.core.FixedAddr,alpha.core.PointerArithm,alpha.core.PointerSub,debug.ExprInspection -analyzer-store=region -verify -triple i686-apple-darwin9 -Wno-tautological-pointer-compare %s

void clang_analyzer_eval(int);

void f1() {
  int a[10];
  int *p = a;
  ++p;
}

char* foo();

void f2() {
  char *p = foo();
  ++p;
}

// This test case checks if we get the right rvalue type of a TypedViewRegion.
// The ElementRegion's type depends on the array region's rvalue type. If it was
// a pointer type, we would get a loc::SymbolVal for '*p'.
void* memchr();
static int
domain_port (const char *domain_b, const char *domain_e,
             const char **domain_e_ptr)
{
  int port = 0;
  
  const char *p;
  const char *colon = memchr (domain_b, ':', domain_e - domain_b);
  
  for (p = colon + 1; p < domain_e ; p++)
    port = 10 * port + (*p - '0');
  return port;
}

void f3() {
  int x, y;
  int d = &y - &x; // expected-warning{{Subtraction of two pointers that do not point to the same memory chunk may cause incorrect result}}

  int a[10];
  int *p = &a[2];
  int *q = &a[8];
  d = q-p; // no-warning
}

void f4() {
  int *p;
  p = (int*) 0x10000; // expected-warning{{Using a fixed address is not portable because that address will probably not be valid in all environments or platforms}}
}

void f5() {
  int x, y;
  int *p;
  p = &x + 1;  // expected-warning{{Pointer arithmetic on non-array variables relies on memory layout, which is dangerous}}

  int a[10];
  p = a + 1; // no-warning
}

// Allow arithmetic on different symbolic regions.
void f6(int *p, int *q) {
  int d = q - p; // no-warning
}

void null_operand(int *a) {
start:
  // LHS is a label, RHS is NULL
  clang_analyzer_eval(&&start != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&&start >= 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&&start > 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((&&start - 0) != 0); // expected-warning{{TRUE}}

  // LHS is a non-symbolic value, RHS is NULL
  clang_analyzer_eval(&a != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a >= 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a > 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((&a - 0) != 0); // expected-warning{{TRUE}}

  // LHS is NULL, RHS is non-symbolic
  // The same code is used for labels and non-symbolic values.
  clang_analyzer_eval(0 != &a); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 <= &a); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 < &a); // expected-warning{{TRUE}}

  // LHS is a symbolic value, RHS is NULL
  clang_analyzer_eval(a != 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a >= 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(a <= 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((a - 0) != 0); // expected-warning{{UNKNOWN}}

  // LHS is NULL, RHS is a symbolic value
  clang_analyzer_eval(0 != a); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(0 <= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 < a); // expected-warning{{UNKNOWN}}
}

void const_locs() {
  char *a = (char*)0x1000;
  char *b = (char*)0x1100;
start:
  clang_analyzer_eval(a != b); // expected-warning{{TRUE}}
  clang_analyzer_eval(a < b); // expected-warning{{TRUE}}
  clang_analyzer_eval(a <= b); // expected-warning{{TRUE}}
  clang_analyzer_eval((b-a) == 0x100); // expected-warning{{TRUE}}

  clang_analyzer_eval(&&start == a); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == &&start); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&a == (char**)a); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((char**)a == &a); // expected-warning{{UNKNOWN}}
}

void array_matching_types() {
  int array[10];
  int *a = &array[2];
  int *b = &array[5];

  clang_analyzer_eval(a != b); // expected-warning{{TRUE}}
  clang_analyzer_eval(a < b); // expected-warning{{TRUE}}
  clang_analyzer_eval(a <= b); // expected-warning{{TRUE}}
  clang_analyzer_eval((b-a) != 0); // expected-warning{{TRUE}}
}

// This takes a different code path than array_matching_types()
void array_different_types() {
  int array[10];
  int *a = &array[2];
  char *b = (char*)&array[5];

  clang_analyzer_eval(a != b); // expected-warning{{TRUE}} expected-warning{{comparison of distinct pointer types}}
  clang_analyzer_eval(a < b); // expected-warning{{TRUE}} expected-warning{{comparison of distinct pointer types}}
  clang_analyzer_eval(a <= b); // expected-warning{{TRUE}} expected-warning{{comparison of distinct pointer types}}
}

struct test { int x; int y; };
void struct_fields() {
  struct test a, b;

  clang_analyzer_eval(&a.x != &a.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a.x < &a.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a.x <= &a.y); // expected-warning{{TRUE}}

  clang_analyzer_eval(&a.x != &b.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a.x > &b.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&a.x >= &b.x); // expected-warning{{UNKNOWN}}
}

void mixed_region_types() {
  struct test s;
  int array[2];
  void *a = &array, *b = &s;

  clang_analyzer_eval(&a != &b); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a > &b); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&a >= &b); // expected-warning{{UNKNOWN}}
}

void symbolic_region(int *p) {
  int a;

  clang_analyzer_eval(&a != p); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a > p); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&a >= p); // expected-warning{{UNKNOWN}}
}

void PR7527 (int *p) {
  if (((int) p) & 1) // not crash
    return;
}

void use_symbols(int *lhs, int *rhs) {
  clang_analyzer_eval(lhs < rhs); // expected-warning{{UNKNOWN}}
  if (lhs < rhs)
    return;
  clang_analyzer_eval(lhs < rhs); // expected-warning{{FALSE}}

  clang_analyzer_eval(lhs - rhs); // expected-warning{{UNKNOWN}}
  if ((lhs - rhs) != 5)
    return;
  clang_analyzer_eval((lhs - rhs) == 5); // expected-warning{{TRUE}}
}

void equal_implies_zero(int *lhs, int *rhs) {
  clang_analyzer_eval(lhs == rhs); // expected-warning{{UNKNOWN}}
  if (lhs == rhs) {
    clang_analyzer_eval(lhs != rhs); // expected-warning{{FALSE}}
    clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{TRUE}}
    return;
  }
  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
  clang_analyzer_eval(lhs != rhs); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{FALSE}}
}

void zero_implies_equal(int *lhs, int *rhs) {
  clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{UNKNOWN}}
  if ((rhs - lhs) == 0) {
    clang_analyzer_eval(lhs != rhs); // expected-warning{{FALSE}}
    clang_analyzer_eval(lhs == rhs); // expected-warning{{TRUE}}
    return;
  }
  clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
  clang_analyzer_eval(lhs != rhs); // expected-warning{{TRUE}}
}

void comparisons_imply_size(int *lhs, int *rhs) {
  clang_analyzer_eval(lhs <= rhs); // expected-warning{{UNKNOWN}}

  if (lhs > rhs) {
    clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{FALSE}}
    return;
  }

  clang_analyzer_eval(lhs <= rhs); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) >= 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{UNKNOWN}}

  if (lhs >= rhs) {
    clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{TRUE}}
    return;
  }

  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
  clang_analyzer_eval(lhs < rhs); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{TRUE}}
}

void size_implies_comparison(int *lhs, int *rhs) {
  clang_analyzer_eval(lhs <= rhs); // expected-warning{{UNKNOWN}}

  if ((rhs - lhs) < 0) {
    clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
    return;
  }

  clang_analyzer_eval(lhs <= rhs); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) >= 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{UNKNOWN}}

  if ((rhs - lhs) <= 0) {
    clang_analyzer_eval(lhs == rhs); // expected-warning{{TRUE}}
    return;
  }

  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
  clang_analyzer_eval(lhs < rhs); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{TRUE}}
}

//-------------------------------
// False positives
//-------------------------------

void zero_implies_reversed_equal(int *lhs, int *rhs) {
  clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{UNKNOWN}}
  if ((rhs - lhs) == 0) {
    // FIXME: Should be FALSE.
    clang_analyzer_eval(rhs != lhs); // expected-warning{{UNKNOWN}}
    // FIXME: Should be TRUE.
    clang_analyzer_eval(rhs == lhs); // expected-warning{{UNKNOWN}}
    return;
  }
  clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{FALSE}}
  // FIXME: Should be FALSE.
  clang_analyzer_eval(rhs == lhs); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(rhs != lhs); // expected-warning{{UNKNOWN}}
}

void canonical_equal(int *lhs, int *rhs) {
  clang_analyzer_eval(lhs == rhs); // expected-warning{{UNKNOWN}}
  if (lhs == rhs) {
    // FIXME: Should be TRUE.
    clang_analyzer_eval(rhs == lhs); // expected-warning{{UNKNOWN}}
    return;
  }
  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}

  // FIXME: Should be FALSE.
  clang_analyzer_eval(rhs == lhs); // expected-warning{{UNKNOWN}}
}

void compare_element_region_and_base(int *p) {
  int *q = p - 1;
  clang_analyzer_eval(p == q); // expected-warning{{FALSE}}
}

struct Point {
  int x;
  int y;
};
void symbolicFieldRegion(struct Point *points, int i, int j) {
  clang_analyzer_eval(&points[i].x == &points[j].x);// expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&points[i].x == &points[i].y);// expected-warning{{FALSE}}
  clang_analyzer_eval(&points[i].x < &points[i].y);// expected-warning{{TRUE}}
}

void negativeIndex(char *str) {
  *(str + 1) = 'a';
  clang_analyzer_eval(*(str + 1) == 'a'); // expected-warning{{TRUE}}
  clang_analyzer_eval(*(str - 1) == 'a'); // expected-warning{{UNKNOWN}}

  char *ptr1 = str - 1;
  clang_analyzer_eval(*ptr1 == 'a'); // expected-warning{{UNKNOWN}}

  char *ptr2 = str;
  ptr2 -= 1;
  clang_analyzer_eval(*ptr2 == 'a'); // expected-warning{{UNKNOWN}}

  char *ptr3 = str;
  --ptr3;
  clang_analyzer_eval(*ptr3 == 'a'); // expected-warning{{UNKNOWN}}
}

