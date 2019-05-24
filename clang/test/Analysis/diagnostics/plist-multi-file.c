// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file -o %t.plist -verify %s
// RUN: tail -n +11 %t.plist | %diff_plist %S/Inputs/expected-plists/plist-multi-file.c.plist -

#include "plist-multi-file.h"

void bar() {
  foo(0);
}
