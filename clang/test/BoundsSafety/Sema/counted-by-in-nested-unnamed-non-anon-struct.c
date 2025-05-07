
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>

struct parent {
  struct {
    union {
      struct {
        int *__counted_by(len) entries;
        int len;
      };
      int *other;
    };
  } s;
};
