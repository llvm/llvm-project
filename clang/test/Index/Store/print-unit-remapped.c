#include "print-unit.h"
#include "syshead.h"

void foo(int i);

// RUN: rm -rf %t
// RUN: %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -fdebug-prefix-map=%S=SRC_ROOT -fdebug-prefix-map=$PWD=BUILD_ROOT -index-store-path %t/idx %s -triple x86_64-apple-macosx10.8
// RUN: c-index-test core -print-unit %t/idx | FileCheck --check-prefixes=ABSOLUTE,ALL %s

// Relative paths should work as well - the unit name, main-path, and out-file should not change.
// RUN: rm -rf %t
// RUN: cd %S && %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -fdebug-prefix-map=%S=SRC_ROOT -index-store-path %t/idx print-unit-remapped.c -o print-unit-remapped.c.o -triple x86_64-apple-macosx10.8
// RUN: c-index-test core -print-unit %t/idx | FileCheck --check-prefixes=RELATIVE,ALL %s

// ALL: print-unit-remapped.c.o-20EK9G967JO97
// ALL: provider: clang-
// ALL: is-system: 0
// ALL: has-main: 1
// ALL: main-path: SRC_ROOT{{/|\\}}print-unit-remapped.c
// ABSOLUTE: work-dir: BUILD_ROOT
// RELATIVE: work-dir: SRC_ROOT
// ALL: out-file: SRC_ROOT{{/|\\}}print-unit-remapped.c.o
// ALL: target: x86_64-apple-macosx10.8
// ALL: is-debug: 1
// ALL: DEPEND START
// ALL: Record | user | SRC_ROOT{{/|\\}}print-unit-remapped.c | print-unit-remapped.c-
// ALL: Record | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}head.h | head.h-
// ALL: Record | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}using-overlay.h | using-overlay.h-
// ALL: Record | system | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h | syshead.h-
// ALL: Record | system | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}another.h | another.h-
// ALL: File | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h{{$}}
// ALL: DEPEND END (6)
// ALL: INCLUDE START
// ALL: SRC_ROOT{{/|\\}}print-unit-remapped.c:1 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h
// ALL: SRC_ROOT{{/|\\}}print-unit-remapped.c:2 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h
// ALL: SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h:1 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}head.h
// ALL: SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h:2 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}using-overlay.h
// ALL: INCLUDE END (4)