#include "print-unit.h"
#include "syshead.h"

void foo(int i);

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cd %t && %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -fdebug-prefix-map=%S/Inputs=INPUT_ROOT -fdebug-prefix-map=%S=SRC_ROOT -fdebug-prefix-map=%t=BUILD_ROOT -index-store-path %t/idx %s -triple x86_64-apple-macosx10.8
// RUN: c-index-test core -print-unit %t/idx | FileCheck --check-prefixes=ABSOLUTE,ALL %s

// Relative paths should work as well - the unit name, main-path, and out-file should not change.
// RUN: rm -rf %t
// RUN: cd %S && %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -fdebug-prefix-map=%S=SRC_ROOT -index-store-path %t/idx print-unit-remapped.c -o print-unit-remapped.c.o -triple x86_64-apple-macosx10.8
// RUN: c-index-test core -print-unit %t/idx | FileCheck --check-prefixes=RELATIVE,ALL %s

// ABSOLUTE: print-unit-remapped.c.o-[[HASH:.+]]
// RELATIVE: print-unit-remapped.c.o-[[HASH]]
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
// ABSOLUTE: Record | user | INPUT_ROOT{{/|\\}}head.h | head.h-
// ABSOLUTE: Record | user | INPUT_ROOT{{/|\\}}using-overlay.h | using-overlay.h-
// ABSOLUTE: Record | system | INPUT_ROOT{{/|\\}}sys{{/|\\}}syshead.h | syshead.h-
// ABSOLUTE: Record | system | INPUT_ROOT{{/|\\}}sys{{/|\\}}another.h | another.h-
// ABSOLUTE: File | user | INPUT_ROOT{{/|\\}}print-unit.h{{$}}
// RELATIVE: Record | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}head.h | head.h-
// RELATIVE: Record | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}using-overlay.h | using-overlay.h-
// RELATIVE: Record | system | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h | syshead.h-
// RELATIVE: Record | system | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}another.h | another.h-
// RELATIVE: File | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h{{$}}
// ALL: DEPEND END (6)
// ALL: INCLUDE START
// ABSOLUTE: SRC_ROOT{{/|\\}}print-unit-remapped.c:1 | INPUT_ROOT{{/|\\}}print-unit.h
// ABSOLUTE: SRC_ROOT{{/|\\}}print-unit-remapped.c:2 | INPUT_ROOT{{/|\\}}sys{{/|\\}}syshead.h
// ABSOLUTE: INPUT_ROOT{{/|\\}}print-unit.h:1 | INPUT_ROOT{{/|\\}}head.h
// ABSOLUTE: INPUT_ROOT{{/|\\}}print-unit.h:2 | INPUT_ROOT{{/|\\}}using-overlay.h
// RELATIVE: SRC_ROOT{{/|\\}}print-unit-remapped.c:1 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h
// RELATIVE: SRC_ROOT{{/|\\}}print-unit-remapped.c:2 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h
// RELATIVE: SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h:1 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}head.h
// RELATIVE: SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h:2 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}using-overlay.h
// ALL: INCLUDE END (4)
