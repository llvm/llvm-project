
// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

typedef struct {
  int len;
  int offs;
  int fam[__counted_by(len - offs)];
} S;

void good(S *s) {
  int arr[10] = {0};
  s = (S *)&arr[5];
  s->offs = 5;
  s->len = 10;
}

void missing_offs(S *s) {
  int arr[10] = {0};
  s = (S *)&arr[5];
  // expected-error@+1{{assignment to 'int' 's->len' requires corresponding assignment to 's->offs'; add self assignment 's->offs = s->offs' if the value has not changed}}
  s->len = 10;
}

void missing_s(S *s) {
  // expected-error@+1{{assignment to 's->offs' requires an immediately preceding assignment to 's' with a wide pointer}}
  s->offs = 5;
  s->len = 10;
}

void missing_len_offs(S *s) {
  int arr[10] = {0};
  s = (S *)&arr[5];
}

void missing_len(S *s) {
  int arr[10] = {0};
  s = (S *)&arr[5];
  // expected-error@+1{{assignment to 'int' 's->offs' requires corresponding assignment to 's->len'; add self assignment 's->len = s->len' if the value has not changed}}
  s->offs = 5;
}

// rdar://132802568
void flexbase_middle(S *s) {
  int arr[10] = {0};
  // expected-error@+1{{assignment to 's->offs' requires an immediately preceding assignment to 's' with a wide pointer}}
  s->offs = 5;
  s = (S *)&arr[5];
  // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
  s->len = 10;
}

void flexbase_last(S *s) {
  int arr[10] = {0};
  // expected-error@+1{{assignment to 's->offs' requires an immediately preceding assignment to 's' with a wide pointer}}
  s->offs = 5;
  s->len = 10;
  s = (S *)&arr[5];
}

typedef struct {
  int len;
  int off1;
  int off2;
  int fam[__counted_by(len - (off1 + off2))];
} S2;

void good2(S2 *s) {
  int arr[10] = {0};
  s = (S2 *)&arr[5];
  s->len = 10;
  s->off1 = 2;
  s->off2 = 3;
}

void missing_off1_off2(S2 *s) {
  int arr[10] = {0};
  s = (S2 *)&arr[5];
  // expected-error@+2{{assignment to 'int' 's->len' requires corresponding assignment to 's->off1'; add self assignment 's->off1 = s->off1' if the value has not changed}}
  // expected-error@+1{{assignment to 'int' 's->len' requires corresponding assignment to 's->off2'; add self assignment 's->off2 = s->off2' if the value has not changed}}
  s->len = 10;
}

void missing_s2(S2 *s) {
  // expected-error@+1{{assignment to 's->off1' requires an immediately preceding assignment to 's' with a wide pointer}}
  s->off1 = 3;
  s->off2 = 2;
  s->len = 10;
}

void missing_len_off1_off2(S2 *s) {
  int arr[10] = {0};
  s = (S2 *)&arr[5];
}


void missing_len_off1(S2 *s) {
  int arr[10] = {0};
  s = (S2 *)&arr[5];
  // expected-error@+2{{assignment to 'int' 's->off2' requires corresponding assignment to 's->off1'; add self assignment 's->off1 = s->off1' if the value has not changed}}
  // expected-error@+1{{assignment to 'int' 's->off2' requires corresponding assignment to 's->len'; add self assignment 's->len = s->len' if the value has not changed}}
  s->off2 = 1;
}

void missing_len_off2(S2 *s) {
  int arr[10] = {0};
  s = (S2 *)&arr[5];
  // expected-error@+2{{assignment to 'int' 's->off1' requires corresponding assignment to 's->off2'; add self assignment 's->off2 = s->off2' if the value has not changed}}
  // expected-error@+1{{assignment to 'int' 's->off1' requires corresponding assignment to 's->len'; add self assignment 's->len = s->len' if the value has not changed}}
  s->off1 = 3;
}

void missing_len2(S2 *s) {
  int arr[10] = {0};
  s = (S2 *)&arr[5];
  // expected-error@+1{{assignment to 'int' 's->off1' requires corresponding assignment to 's->len'; add self assignment 's->len = s->len' if the value has not changed}}
  s->off1 = 2;
  s->off2 = 3;
}

void missing_len_off1_2(S2 *s) {
  int arr[10] = {0};
  s = (S2 *)&arr[5];
}

// rdar://132802568
void flexbase_middle2(S2 *s) {
  int arr[10] = {0};
  // expected-error@+1{{assignment to 's->len' requires an immediately preceding assignment to 's' with a wide pointer}}
  s->len = 10;
  s->off1 = 2;
  s = (S2 *)&arr[5];
  // expected-error@+1{{assignments to dependent variables should not have side effects between them}}
  s->off2 = 3;
}

void flexbase_last2(S2 *s) {
  int arr[10] = {0};
  // expected-error@+1{{assignment to 's->len' requires an immediately preceding assignment to 's' with a wide pointer}}
  s->len = 10;
  s->off1 = 2;
  s->off2 = 3;
  s = (S2 *)&arr[5];
}
