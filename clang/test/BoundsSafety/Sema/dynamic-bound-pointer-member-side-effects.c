
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct CountedByStruct {
  int len;
  int *__counted_by(len) buf;
};

struct EndedByStruct {
  int *end;
  int *__ended_by(end) iter;
  int *__ended_by(iter) start;
};

struct CountedByStruct RetCountedByVal(void);
struct CountedByStruct *RetCountedByPtr(void);
struct EndedByStruct RetEndedByVal(void);
struct EndedByStruct *RetEndedByPtr(void);

void Test(void) {
  int *b1 = RetCountedByVal().buf; // ok
  int *b2 = RetCountedByPtr()->buf; // ok
  int l1 = RetCountedByVal().len; // ok
  int l2 = RetCountedByPtr()->len; // ok

  int *b3 = RetEndedByVal().end; // ok
  int *b4 = RetEndedByVal().iter; // ok
  int *b5 = RetEndedByVal().start; // ok

  int *b6 = RetEndedByPtr()->end; // ok
  int *b7 = RetEndedByPtr()->iter; // ok
  int *b8 = RetEndedByPtr()->start; // ok

}

// expected-no-diagnostics
