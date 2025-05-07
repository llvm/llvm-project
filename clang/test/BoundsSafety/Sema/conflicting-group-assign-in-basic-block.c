
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct CountedByData {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

void TestCountedBy(struct CountedByData *d) {
  int arr[10];
  d->bp = &arr[1]; // expected-note{{previously assigned here}}
  d->bp2 = arr; // expected-note{{previously assigned here}}
  d->l = 9; // expected-note{{previously assigned here}}

  // expected-error@+1{{multiple consecutive assignments to a dynamic count pointer 'bp' must be simplified; keep only one of the assignments}}
  d->bp = d->bp;
  // expected-error@+1{{multiple consecutive assignments to a dynamic count pointer 'bp2' must be simplified; keep only one of the assignments}}
  d->bp2 = &arr[0];
  // expected-error@+1{{multiple consecutive assignments to a dynamic count 'l' must be simplified; keep only one of the assignments}}
  d->l = d->l;
}

void AnyCall(void);

void TestCountedByCallInBetween(struct CountedByData *d) {
  int arr[10];
  d->bp = &arr[1];
  d->bp2 = arr;
  d->l = 9;

  AnyCall();

  d->bp = d->bp;
  d->bp2 = &arr[0];
  d->l = d->l;
}

void TestCountedByMultipleInstances(struct CountedByData *d1, struct CountedByData *d2) {
  int arr[10];
  d1->bp = &arr[1];
  d1->bp2 = arr;
  d1->l = 9;

  d2->bp = d1->bp;
  d2->bp2 = &arr[0];
  d2->l = d1->l;
}

struct EndedByData {
    int *__ended_by(iter) start;
    int *__ended_by(end) iter;
    int *end;
};

void TestEndedBy(struct EndedByData *d) {
  int arr[10];
  d->start = arr + 1; // expected-note{{previously assigned here}}
  // expected-error@+1{{multiple consecutive assignments to a ranged pointer 'start' must be simplified; keep only one of the assignments}}
  d->start = arr;
  d->iter = arr; // expected-note{{previously assigned here}}
  d->end = arr + 10; // expected-note{{previously assigned here}}
  // expected-error@+1{{multiple consecutive assignments to a ranged pointer 'end' must be simplified; keep only one of the assignments}}
  d->end = &arr[0] + 5;
  // expected-error@+1{{multiple consecutive assignments to a ranged pointer 'iter' must be simplified; keep only one of the assignments}}
  d->iter = &arr[0] + 6;
}
