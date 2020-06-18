// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-checker=debug.ExprInspection -verify %s

typedef unsigned long size_t;
size_t clang_analyzer_getExtent(void *);
void clang_analyzer_eval(int);

// Zero-sized VLAs.
void check_zero_sized_VLA(int x) {
  if (x)
    return;

  int vla[x]; // expected-warning{{Declared variable-length array (VLA) has zero size}}
}

void check_uninit_sized_VLA() {
  int x;
  int vla[x]; // expected-warning{{Declared variable-length array (VLA) uses a garbage value as its size}}
}

// Negative VLAs.
static void vla_allocate_signed(short x) {
  int vla[x]; // expected-warning{{Declared variable-length array (VLA) has negative size}}
}

static void vla_allocate_unsigned(unsigned short x) {
  int vla[x]; // no-warning
}

void check_negative_sized_VLA_1() {
  vla_allocate_signed(-1);
}

void check_negative_sized_VLA_2() {
  vla_allocate_unsigned(-1);
}

void check_negative_sized_VLA_3() {
  short x = -1;
  int vla[x]; // expected-warning{{Declared variable-length array (VLA) has negative size}}
}

void check_negative_sized_VLA_4() {
  unsigned short x = -1;
  int vla[x]; // no-warning
}

void check_negative_sized_VLA_5() {
  signed char x = -1;
  int vla[x]; // expected-warning{{Declared variable-length array (VLA) has negative size}}
}

void check_negative_sized_VLA_6() {
  unsigned char x = -1;
  int vla[x]; // no-warning
}

void check_negative_sized_VLA_7() {
  signed char x = -1;
  int vla[x + 2]; // no-warning
}

void check_negative_sized_VLA_8() {
  signed char x = 1;
  int vla[x - 2]; // expected-warning{{Declared variable-length array (VLA) has negative size}}
}

void check_negative_sized_VLA_9() {
  int x = 1;
  int vla[x]; // no-warning
}

static void check_negative_sized_VLA_10_sub(int x)
{
  int vla[x]; // expected-warning{{Declared variable-length array (VLA) has negative size}}
}

void check_negative_sized_VLA_10(int x) {
  if (x < 0)
    check_negative_sized_VLA_10_sub(x);
}

static void check_negative_sized_VLA_11_sub(short x)
{
  int vla[x]; // no-warning
}

void check_negative_sized_VLA_11(short x) {
  if (x > 0)
    check_negative_sized_VLA_11_sub(x);
}

void check_VLA_typedef() {
  int x = -1;
  typedef int VLA[x]; // expected-warning{{Declared variable-length array (VLA) has negative size}}
}

size_t check_VLA_sizeof() {
  int x = -1;
  size_t s = sizeof(int[x]); // expected-warning{{Declared variable-length array (VLA) has negative size}}
  return s;
}

// Multi-dimensional arrays.

void check_zero_sized_VLA_multi1(int x) {
  if (x)
    return;

  int vla[10][x]; // expected-warning{{Declared variable-length array (VLA) has zero size}}
}

void check_zero_sized_VLA_multi2(int x, int y) {
  if (x)
    return;

  int vla[y][x]; // expected-warning{{Declared variable-length array (VLA) has zero size}}
}

// Check the extent.

void check_VLA_extent() {
  int x = 3;

  int vla1[x];
  clang_analyzer_eval(clang_analyzer_getExtent(&vla1) == x * sizeof(int));
  // expected-warning@-1{{TRUE}}

  int vla2[x][2];
  clang_analyzer_eval(clang_analyzer_getExtent(&vla2) == x * 2 * sizeof(int));
  // expected-warning@-1{{TRUE}}

  int vla2m[2][x];
  clang_analyzer_eval(clang_analyzer_getExtent(&vla2m) == 2 * x * sizeof(int));
  // expected-warning@-1{{TRUE}}

  int vla3m[2][x][4];
  clang_analyzer_eval(clang_analyzer_getExtent(&vla3m) == 2 * x * 4 * sizeof(int));
  // expected-warning@-1{{TRUE}}
}

// https://bugs.llvm.org/show_bug.cgi?id=46128
// analyzer doesn't handle more than simple symbolic expressions.
// Just don't crash.
extern void foo(void);
int a;
void b() {
  int c = a + 1;
  for (;;) {
    int d[c];
    for (; 0 < c;)
      foo();
  }
} // no-crash
