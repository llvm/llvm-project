// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.PointerSub -analyzer-output=text -verify %s

void negative_1() {
  int a[3];
  int x = -1;
  // FIXME: should indicate that 'x' is -1
  int d = &a[x] - &a[0]; // expected-warning{{Using a negative array index at pointer subtraction is undefined behavior}} \
                         // expected-note{{Using a negative array index at pointer subtraction is undefined behavior}}
}

void negative_2() {
  int a[3];
  int *p1 = a, *p2 = a;
  --p2;
  // FIXME: should indicate that 'p2' is negative
  int d = p1 - p2; // expected-warning{{Using a negative array index at pointer subtraction is undefined behavior}} \
                   // expected-note{{Using a negative array index at pointer subtraction is undefined behavior}}
}

void different_1() {
  int a[3]; // expected-note{{Array at the left-hand side of subtraction}}
  int b[3]; // expected-note{{Array at the right-hand side of subtraction}}
  int d = &a[2] - &b[0]; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}} \
                         // expected-note{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
}

void different_2() {
  int a[3]; // expected-note{{Array at the right-hand side of subtraction}}
  int b[3]; // expected-note{{Array at the left-hand side of subtraction}}
  int *p1 = a + 1;
  int *p2 = b;
  int d = p2 - p1; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}} \
                   // expected-note{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
}

int different_3() {
  struct {
    int array[5];
  } a, b;
  return &a.array[3] - &b.array[2]; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}} \
                                    // expected-note{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
}

int different_4() {
  struct {
    int array1[5]; // expected-note{{Array at the left-hand side of subtraction}}
    int array2[5]; // expected-note{{Array at the right-hand side of subtraction}}
  } a;
  return &a.array1[3] - &a.array2[4]; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}} \
                                      // expected-note{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
}

void different_5() {
  int d;
  static int x[10][10]; // expected-note2{{Array at the left-hand side of subtraction}}
  int *y1 = &(x[3][5]);
  char *z = ((char *) y1) + 2;
  int *y2 = (int *)(z - 2);
  int *y3 = ((int *)x) + 35; // This is offset for [3][5].

  d = y2 - y1; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}} \
               // expected-note{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  d = y3 - y1; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}} \
               // expected-note{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  d = y3 - y2;
}
