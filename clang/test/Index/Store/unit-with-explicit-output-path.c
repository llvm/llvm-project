#include "print-unit.h"
#include "syshead.h"

void foo(int i);

// RUN: rm -rf %t
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx %s -triple x86_64-apple-macosx10.8
// RUN: c-index-test core -print-unit %t/idx | FileCheck --check-prefixes=DEFAULT,ALL %s
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx_opt1 %s -triple x86_64-apple-macosx10.8 -o %t/unit-with-explicit-output-path.o
// RUN: c-index-test core -print-unit %t/idx_opt1 | FileCheck %s -check-prefixes=OUT,ALL
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx_opt2 %s -triple x86_64-apple-macosx10.8 -index-unit-output-path custom-unit.o
// RUN: c-index-test core -print-unit %t/idx_opt2 | FileCheck %s -check-prefixes=UNITOUT,ALL

// DEFAULT: unit-with-explicit-output-path.c.o
// OUT: unit-with-explicit-output-path.o
// UNITOUT: custom-unit.o
// ALL: provider: clang-
// ALL: is-system: 0
// ALL: has-main: 1
// ALL: main-path: {{.*}}{{/|\\}}unit-with-explicit-output-path.c
// DEFAULT: out-file: {{.*}}{{/|\\}}unit-with-explicit-output-path.c.o
// OUT: out-file: {{.*}}{{/|\\}}unit-with-explicit-output-path.o
// UNITOUT: out-file: {{.*}}{{/|\\}}custom-unit.o
// ALL: target: x86_64-apple-macosx10.8
// ALL: is-debug: 1


// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx_same %s -triple x86_64-apple-macosx10.8 -o %t/dir1/out.o -index-unit-output-path %t/custom-unit.o
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx_same %s -triple x86_64-apple-macosx10.8 -o %t/dir2/out.o -index-unit-output-path %t/custom-unit.o
// RUN: c-index-test core -print-unit %t/idx_same | FileCheck %s -check-prefixes=SINGLE

// Make sure there's only one unit file produced even though both invocations had different -o paths.
// SINGLE-NOT: --------
// SINGLE: custom-unit.o-{{.*}}
// SINGLE-NEXT: --------
// SINGLE-NOT: --------

// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -index-store-path %t/idx_same %s -triple x86_64-apple-macosx10.8 -o %t/dir2/out.o -index-unit-output-path %t/custom-unit2.o
// RUN: c-index-test core -print-unit %t/idx_same | FileCheck %s -check-prefixes=TWOUNITS

// Make sure there are two unit files now, as we had a different unit output path.

// TWOUNITS: custom-unit.o-{{.*}}
// TWOUNITS-NEXT: --------
// TWOUNITS: custom-unit2.o-{{.*}}
// TWOUNITS-NEXT: --------