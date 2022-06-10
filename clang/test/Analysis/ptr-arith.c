// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.FixedAddr,alpha.core.PointerArithm,alpha.core.PointerSub,debug.ExprInspection -Wno-pointer-to-int-cast -verify -triple x86_64-apple-darwin9 -Wno-tautological-pointer-compare -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.FixedAddr,alpha.core.PointerArithm,alpha.core.PointerSub,debug.ExprInspection -Wno-pointer-to-int-cast -verify -triple i686-apple-darwin9 -Wno-tautological-pointer-compare -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);

void f1(void) {
  int a[10];
  int *p = a;
  ++p;
}

char* foo(void);

void f2(void) {
  char *p = foo();
  ++p;
}

// This test case checks if we get the right rvalue type of a TypedViewRegion.
// The ElementRegion's type depends on the array region's rvalue type. If it was
// a pointer type, we would get a loc::SymbolVal for '*p'.
void* memchr(const void *, int, __typeof__(sizeof(0)));
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

void f3(void) {
  int x, y;
  int d = &y - &x; // expected-warning{{Subtraction of two pointers that do not point to the same memory chunk may cause incorrect result}}

  int a[10];
  int *p = &a[2];
  int *q = &a[8];
  d = q-p; // no-warning
}

void f4(void) {
  int *p;
  p = (int*) 0x10000; // expected-warning{{Using a fixed address is not portable because that address will probably not be valid in all environments or platforms}}
}

void f5(void) {
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

void const_locs(void) {
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

void array_matching_types(void) {
  int array[10];
  int *a = &array[2];
  int *b = &array[5];

  clang_analyzer_eval(a != b); // expected-warning{{TRUE}}
  clang_analyzer_eval(a < b); // expected-warning{{TRUE}}
  clang_analyzer_eval(a <= b); // expected-warning{{TRUE}}
  clang_analyzer_eval((b-a) != 0); // expected-warning{{TRUE}}
}

// This takes a different code path than array_matching_types()
void array_different_types(void) {
  int array[10];
  int *a = &array[2];
  char *b = (char*)&array[5];

  clang_analyzer_eval(a != b); // expected-warning{{TRUE}} expected-warning{{comparison of distinct pointer types}}
  clang_analyzer_eval(a < b); // expected-warning{{TRUE}} expected-warning{{comparison of distinct pointer types}}
  clang_analyzer_eval(a <= b); // expected-warning{{TRUE}} expected-warning{{comparison of distinct pointer types}}
}

struct test { int x; int y; };
void struct_fields(void) {
  struct test a, b;

  clang_analyzer_eval(&a.x != &a.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a.x < &a.y); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a.x <= &a.y); // expected-warning{{TRUE}}

  clang_analyzer_eval(&a.x != &b.x); // expected-warning{{TRUE}}
  clang_analyzer_eval(&a.x > &b.x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(&a.x >= &b.x); // expected-warning{{UNKNOWN}}
}

void mixed_region_types(void) {
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
// FIXME: In Z3ConstraintManager, ptrdiff_t is mapped to signed bitvector. However, this does not directly imply the unsigned comparison.
#ifdef ANALYZER_CM_Z3
  clang_analyzer_eval((rhs - lhs) >= 0); // expected-warning{{UNKNOWN}}
#else
  clang_analyzer_eval((rhs - lhs) >= 0); // expected-warning{{TRUE}}
#endif
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{UNKNOWN}}

  if (lhs >= rhs) {
    clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{TRUE}}
    return;
  }

  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
  clang_analyzer_eval(lhs < rhs); // expected-warning{{TRUE}}
#ifdef ANALYZER_CM_Z3
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{UNKNOWN}}
#else
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{TRUE}}
#endif
}

void size_implies_comparison(int *lhs, int *rhs) {
  clang_analyzer_eval(lhs <= rhs); // expected-warning{{UNKNOWN}}

  if ((rhs - lhs) < 0) {
    clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
    return;
  }

#ifdef ANALYZER_CM_Z3
  clang_analyzer_eval(lhs <= rhs); // expected-warning{{UNKNOWN}}
#else
  clang_analyzer_eval(lhs <= rhs); // expected-warning{{TRUE}}
#endif
  clang_analyzer_eval((rhs - lhs) >= 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{UNKNOWN}}

  if ((rhs - lhs) <= 0) {
    clang_analyzer_eval(lhs == rhs); // expected-warning{{TRUE}}
    return;
  }

  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
#ifdef ANALYZER_CM_Z3
  clang_analyzer_eval(lhs < rhs); // expected-warning{{UNKNOWN}}
#else
  clang_analyzer_eval(lhs < rhs); // expected-warning{{TRUE}}
#endif
  clang_analyzer_eval((rhs - lhs) > 0); // expected-warning{{TRUE}}
}

void zero_implies_reversed_equal(int *lhs, int *rhs) {
  clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{UNKNOWN}}
  if ((rhs - lhs) == 0) {
    clang_analyzer_eval(rhs != lhs); // expected-warning{{FALSE}}
    clang_analyzer_eval(rhs == lhs); // expected-warning{{TRUE}}
    return;
  }
  clang_analyzer_eval((rhs - lhs) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(rhs == lhs); // expected-warning{{FALSE}}
  clang_analyzer_eval(rhs != lhs); // expected-warning{{TRUE}}
}

void canonical_equal(int *lhs, int *rhs) {
  clang_analyzer_eval(lhs == rhs); // expected-warning{{UNKNOWN}}
  if (lhs == rhs) {
    clang_analyzer_eval(rhs == lhs); // expected-warning{{TRUE}}
    return;
  }
  clang_analyzer_eval(lhs == rhs); // expected-warning{{FALSE}}
  clang_analyzer_eval(rhs == lhs); // expected-warning{{FALSE}}
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

void test_no_crash_on_pointer_to_label(void) {
  char *a = &&label;
  a[0] = 0;
label:;
}

typedef __attribute__((__ext_vector_type__(2))) float simd_float2;
float test_nowarning_on_vector_deref(void) {
  simd_float2 x = {0, 1};
  return x[1]; // no-warning
}

struct s {
  int v;
};

// These three expressions should produce the same sym vals.
void struct_pointer_canon(struct s *ps) {
  struct s ss = *ps;
  clang_analyzer_dump((*ps).v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<struct s * ps>}.v>}}
  clang_analyzer_dump(ps[0].v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<struct s * ps>}.v>}}
  clang_analyzer_dump(ps->v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<struct s * ps>}.v>}}
  clang_analyzer_eval((*ps).v == ps[0].v); // expected-warning{{TRUE}}
  clang_analyzer_eval((*ps).v == ps->v);   // expected-warning{{TRUE}}
  clang_analyzer_eval(ps[0].v == ps->v);   // expected-warning{{TRUE}}
}

void struct_pointer_canon_bidim(struct s **ps) {
  struct s ss = **ps;
  clang_analyzer_eval(&(ps[0][0].v) == &((*ps)->v)); // expected-warning{{TRUE}}
}

typedef struct s T1;
typedef struct s T2;
void struct_pointer_canon_typedef(T1 *ps) {
  T2 ss = *ps;
  clang_analyzer_dump((*ps).v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<T1 * ps>}.v>}}
  clang_analyzer_dump(ps[0].v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<T1 * ps>}.v>}}
  clang_analyzer_dump(ps->v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<T1 * ps>}.v>}}
  clang_analyzer_eval((*ps).v == ps[0].v); // expected-warning{{TRUE}}
  clang_analyzer_eval((*ps).v == ps->v);   // expected-warning{{TRUE}}
  clang_analyzer_eval(ps[0].v == ps->v);   // expected-warning{{TRUE}}
}

void struct_pointer_canon_bidim_typedef(T1 **ps) {
  T2 ss = **ps;
  clang_analyzer_eval(&(ps[0][0].v) == &((*ps)->v)); // expected-warning{{TRUE}}
}

void struct_pointer_canon_const(const struct s *ps) {
  struct s ss = *ps;
  clang_analyzer_dump((*ps).v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<const struct s * ps>}.v>}}
  clang_analyzer_dump(ps[0].v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<const struct s * ps>}.v>}}
  clang_analyzer_dump(ps->v);
  // expected-warning-re@-1{{reg_${{[[:digit:]]+}}<int SymRegion{reg_${{[[:digit:]]+}}<const struct s * ps>}.v>}}
  clang_analyzer_eval((*ps).v == ps[0].v); // expected-warning{{TRUE}}
  clang_analyzer_eval((*ps).v == ps->v);   // expected-warning{{TRUE}}
  clang_analyzer_eval(ps[0].v == ps->v);   // expected-warning{{TRUE}}
}
