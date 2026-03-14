// Test that diagnostic mappings are emitted only when needed and in order of
// diagnostic ID rather than non-deterministically. This test passes 3
// -W options and expects exactly 3 mappings to be emitted in the pcm. The -W
// options are chosen to be far apart in ID (see DiagnosticIDs.h) so we can
// check they are ordered. We also intentionally trigger several other warnings
// inside the module and ensure they do not show up in the pcm as mappings.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -triple x86_64-apple-macosx10.11.0 \
// RUN:   %t/main.m -fdisable-module-hash \
// RUN:   -Werror=stack-protector -Werror=empty-translation-unit -Werror=float-equal

// RUN: mv %t/cache/A.pcm %t/A1.pcm

// RUN: llvm-bcanalyzer --dump --disable-histogram %t/A1.pcm | FileCheck %s

// CHECK: <DIAG_PRAGMA_MAPPINGS

// == Initial mappings
// Number of mappings = 3
// CHECK-SAME: op2=3
// Common diag id is < 1000 (see DiagnosticIDs.h)
// CHECK-SAME: op3=[[STACK_PROT:[0-9][0-9]?[0-9]?]] op4=
// Parse diag id is somewhere in 1000..2999, leaving room for changes
// CHECK-SAME: op5=[[EMPTY_TU:[12][0-9][0-9][0-9]]] op6=
// Sema diag id is > 2000
// CHECK-SAME: op7=[[FLOAT_EQ:[2-9][0-9][0-9][0-9]]] op8=

// == Pragmas:
// Each pragma creates a mapping table; and each copies the previous table. The
// initial mappings are copied as well, but are not serialized since they have
// isPragma=false.

// == ignored "-Wfloat-equal"
// CHECK-SAME: op{{[0-9]+}}=1
// CHECK-SAME: op{{[0-9]+}}=[[FLOAT_EQ]] op{{[0-9]+}}=

// == ignored "-Wstack-protector"
// CHECK-SAME: op{{[0-9]+}}=2
// CHECK-SAME: op{{[0-9]+}}=[[STACK_PROT]] op{{[0-9]+}}=
// CHECK-SAME: op{{[0-9]+}}=[[FLOAT_EQ]] op{{[0-9]+}}=

// == warning "-Wempty-translation-unit"
// CHECK-SAME: op{{[0-9]+}}=3
// CHECK-SAME: op{{[0-9]+}}=[[STACK_PROT]] op{{[0-9]+}}=
// CHECK-SAME: op{{[0-9]+}}=[[EMPTY_TU]] op{{[0-9]+}}=
// CHECK-SAME: op{{[0-9]+}}=[[FLOAT_EQ]] op{{[0-9]+}}=

// == warning "-Wstack-protector"
// CHECK-SAME: op{{[0-9]+}}=3
// CHECK-SAME: op{{[0-9]+}}=[[STACK_PROT]] op{{[0-9]+}}=
// CHECK-SAME: op{{[0-9]+}}=[[EMPTY_TU]] op{{[0-9]+}}=
// CHECK-SAME: op{{[0-9]+}}=[[FLOAT_EQ]] op{{[0-9]+}}=

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -triple x86_64-apple-macosx10.11.0 \
// RUN:   %t/main.m -fdisable-module-hash \
// RUN:   -Werror=stack-protector -Werror=empty-translation-unit -Werror=float-equal

// RUN: diff %t/cache/A.pcm %t/A1.pcm

//--- module.modulemap
module A { header "a.h" }

//--- a.h
// Lex warning
#warning "w"

static inline void f() {
// Parse warning
  ;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wstack-protector"

static inline void g() {
// Sema warning
  int x;
}

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wempty-translation-unit"
#pragma clang diagnostic warning "-Wstack-protector"

#pragma clang diagnostic pop
#pragma clang diagnostic pop

//--- main.m
#import "a.h"
