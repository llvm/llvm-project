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
