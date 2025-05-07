// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

union S {
  int *p1 __counted_by(count);
  // expected-error@-1 {{cannot use '__counted_by' on union fields}}
  int count;
  int *p2 __ended_by(end);
  // expected-error@-1 {{cannot use '__ended_by' on union fields}}
  int *end;
  int *p3 __sized_by(size);
  // expected-error@-1 {{cannot use '__sized_by' on union fields}}
  int size;
  struct
  {
    int *p1 __counted_by(count);
    int count;
    int *p2 __ended_by(end);
    int *end;
    int *p3 __sized_by(size);
    int size;
  } nested;

  struct
  {
    int *p4 __counted_by(count2);
    int count2;
    int *p5 __ended_by(end2);
    int *end2;
    int *p6 __sized_by(size2);
    int size2;
  } /* anonymous */;
};
