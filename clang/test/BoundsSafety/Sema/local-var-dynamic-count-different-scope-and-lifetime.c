

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

int g_len1;

void f1(void) {
  // expected-error@+2{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
  int *__counted_by(g_len1) ptr;
  // expected-error@+2{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
  int *__sized_by(g_len1) ptr2;
  // expected-error@+2{{argument of '__counted_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  int *__counted_by_or_null(g_len1) ptr3;
  // expected-error@+2{{argument of '__sized_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  int *__sized_by_or_null(g_len1) ptr4;
}

static int g_len2;

void f2(void) {
  // expected-error@+2{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
  int *__counted_by(g_len2) ptr;
  // expected-error@+2{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
  int *__sized_by(g_len2) ptr2;
  // expected-error@+2{{argument of '__counted_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  int *__counted_by_or_null(g_len2) ptr3;
  // expected-error@+2{{argument of '__sized_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  int *__sized_by_or_null(g_len2) ptr4;
}

int g_len3;

void f3(void) {
  // expected-error@+2{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
  static int *__counted_by(g_len3) ptr;
  // expected-error@+2{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
  static int *__sized_by(g_len3) ptr2;
  // expected-error@+2{{argument of '__counted_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  static int *__counted_by_or_null(g_len3) ptr3;
  // expected-error@+2{{argument of '__sized_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  static int *__sized_by_or_null(g_len3) ptr4;
}

static int g_len4;

void f4(void) {
  // expected-error@+2{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
  static int *__counted_by(g_len4) ptr;
  // expected-error@+2{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
  static int *__sized_by(g_len4) ptr2;
  // expected-error@+2{{argument of '__counted_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  static int *__counted_by_or_null(g_len4) ptr3;
  // expected-error@+2{{argument of '__sized_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by_or_null' attribute cannot refer to declaration of a different lifetime}}
  static int *__sized_by_or_null(g_len4) ptr4;
}

void f5(void) {
  static int len;
  {
    // expected-error@+2{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
    // expected-error@+1{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
    int *__counted_by(len) ptr;
  // expected-error@+2{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
    int *__sized_by(len) ptr2;
  // expected-error@+2{{argument of '__counted_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by_or_null' attribute cannot refer to declaration of a different lifetime}}
    int *__counted_by_or_null(len) ptr3;
  // expected-error@+2{{argument of '__sized_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by_or_null' attribute cannot refer to declaration of a different lifetime}}
    int *__sized_by_or_null(len) ptr4;
  }
}

void f6(void) {
  int len;
  {
    // expected-error@+2{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
    // expected-error@+1{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
    static int *__counted_by(len) ptr;
  // expected-error@+2{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
    static int *__sized_by(len) ptr2;
  // expected-error@+2{{argument of '__counted_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by_or_null' attribute cannot refer to declaration of a different lifetime}}
    static int *__counted_by_or_null(len) ptr3;
  // expected-error@+2{{argument of '__sized_by_or_null' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__sized_by_or_null' attribute cannot refer to declaration of a different lifetime}}
    static int *__sized_by_or_null(len) ptr4;
  }
}
