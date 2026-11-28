// RUN: %check_clang_tidy -std=c++14-or-later \
// RUN:   -check-header %S/Inputs/macro-to-enum/expansion-uses.h \
// RUN:   -check-header %S/Inputs/macro-to-enum/expansion-local.h \
// RUN:   -check-header %S/Inputs/macro-to-enum/expansion-redefined.h \
// RUN:   %s modernize-macro-to-enum %t -- \
// RUN:   -- -I%S/Inputs/macro-to-enum -Wno-macro-redefined -fno-delayed-template-parsing
// RUN: %clang -std=c++14 -Wno-macro-redefined -fsyntax-only -I%t.cpp.headers %t.cpp

void use_one(int);

#include "expansion-uses.h"
#include "expansion-uses.h"

// The definition is in a header, but its scope is the enclosing function.
void local_header_definition() {
#include "expansion-local.h"
}

int LocalUse = HEADER_LOCAL;

void old_header_definition() {
#define REDEFINED_HEADER (21)
  // CHECK-FIXES: #define REDEFINED_HEADER (21)
}

#include "expansion-redefined.h"

// Keep a positive control so source diagnostics are checked as well.
#define SOURCE_SAFE (23)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'SOURCE_SAFE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: SOURCE_SAFE = (23)
// CHECK-FIXES-NEXT: };

void including_file_uses() {
  use_one(SOURCE_SAFE);
  use_one(HEADER_SAFE);
  use_one(HEADER_SAFE);
  use_one HEADER_LATE_BAD;
  use_one REDEFINED_HEADER;
}
