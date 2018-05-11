// RUN: mkdir %T/Inputs
// RUN: cp -r %S/Inputs/fuchsia-restrict-system-includes %T/Inputs/fuchsia-restrict-system-includes
// RUN: %check_clang_tidy %s fuchsia-restrict-system-includes %t \
// RUN:		-- -config="{CheckOptions: [{key: fuchsia-restrict-system-includes.Includes, value: 'transitive.h,s.h'}]}" \
// RUN:   -system-headers -header-filter=.* \
// RUN:   -- -std=c++11 -I %T/Inputs/fuchsia-restrict-system-includes -isystem %T/Inputs/fuchsia-restrict-system-includes/system
// RUN: FileCheck -input-file=%T/Inputs/fuchsia-restrict-system-includes/transitive2.h %s -check-prefix=CHECK-FIXES
// RUN: rm -rf %T/Inputs

// transitive.h includes <r.h> and <t.h>
#include <transitive.h>
// CHECK-MESSAGES: :1:1: warning: system include r.h not allowed, transitively included from {{(.*\/)*}}Inputs/fuchsia-restrict-system-includes/system/transitive.h
// CHECK-MESSAGES: :2:1: warning: system include t.h not allowed, transitively included from {{(.*\/)*}}Inputs/fuchsia-restrict-system-includes/system/transitive.h

// transitive.h includes <s.h> and <t.h>
#include "transitive2.h"
// CHECK-MESSAGES: :2:1: warning: system include t.h not allowed, transitively included from {{(.*\/)*}}Inputs/fuchsia-restrict-system-includes/transitive2.h
// CHECK-FIXES-NOT: #include <t.h>

int main() {
  // f() is declared in r.h
}
