// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-html -o %t.plist -verify %s
// RUN: tail -n +11 %t.plist | %diff_plist --ignore-matching-lines=report %S/Inputs/expected-plists/plist-multi-file.c.plist -

#include "plist-multi-file.h"

void bar() {
  foo(0);
}
