
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct flex {
  int len;
  char fam[];
};

struct flex_with_count {
  int len;
  char fam[__counted_by(len)];
};

struct flex_in_union {
  struct flex fnc; // expected-warning{{field 'fnc' with variable sized type 'struct flex' not at the end of a struct or class is a GNU extension}}
  union _u{
    struct flex_with_count fc;
    int i;
  } u; // expected-warning{{field 'u' with variable sized type 'union _u' not at the end of a struct or class is a GNU extension}}

  int dummy;
};

char foo(struct flex_in_union * bar) {
  // expected-warning@+1{{accessing elements of an unannotated incomplete array always fails at runtime}}
  return bar->dummy ? bar->fnc.fam[bar->fnc.len - 1] : bar->u.fc.fam[bar->u.fc.len - 1];
}
