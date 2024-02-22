#include "print-unit.h"
#include "syshead.h"

void foo(int i);

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cd %t && %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -fdebug-prefix-map=%S=SRC_ROOT -fdebug-prefix-map=%t=BUILD_ROOT -index-store-path %t/idx %s -triple x86_64-apple-macosx10.8
// RUN: c-index-test core -print-unit %t/idx -index-store-prefix-map SRC_ROOT=%S -index-store-prefix-map BUILD_ROOT=$PWD | FileCheck %s

// CHECK: print-unit-roundtrip-remapping.c.o-{{.+}}
// CHECK: provider: clang-
// CHECK: is-system: 0
// CHECK: has-main: 1
// CHECK-NOT: main-path: SRC_ROOT{{/|\\}}print-unit-roundtrip-remapping.c
// CHECK: main-path: {{.*}}{{/|\\}}print-unit-roundtrip-remapping.c
// CHECK-NOT: work-dir: BUILD_ROOT
// CHECK: out-file: SRC_ROOT{{/|\\}}print-unit-roundtrip-remapping.c.o
// CHECK: target: x86_64-apple-macosx10.8
// CHECK: is-debug: 1
// CHECK: DEPEND START
// CHECK-NOT: Record | user | SRC_ROOT{{/|\\}}print-unit-roundtrip-remapping.c | print-unit-roundtrip-remapping.c-
// CHECK: Record | user | {{.*}}{{/|\\}}print-unit-roundtrip-remapping.c | print-unit-roundtrip-remapping.c-
// CHECK-NOT: Record | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}head.h | head.h-
// CHECK: Record | user | {{.*}}{{/|\\}}Inputs{{/|\\}}head.h | head.h-
// CHECK-NOT: Record | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}using-overlay.h | using-overlay.h-
// CHECK: Record | user | {{.*}}{{/|\\}}Inputs{{/|\\}}using-overlay.h | using-overlay.h-
// CHECK-NOT: Record | system | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h | syshead.h-
// CHECK: Record | system | {{.*}}{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h | syshead.h-
// CHECK-NOT: Record | system | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}another.h | another.h-
// CHECK: Record | system | {{.*}}{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}another.h | another.h-
// CHECK-NOT: File | user | SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h{{$}}
// CHECK: File | user | {{.*}}{{/|\\}}Inputs{{/|\\}}print-unit.h{{$}}
// CHECK: DEPEND END (6)
// CHECK: INCLUDE START
// CHECK-NOT: SRC_ROOT{{/|\\}}print-unit-roundtrip-remapping.c:1 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h
// CHECK: {{.*}}{{/|\\}}print-unit-roundtrip-remapping.c:1 | {{.*}}{{/|\\}}Inputs{{/|\\}}print-unit.h
// CHECK-NOT: SRC_ROOT{{/|\\}}print-unit-roundtrip-remapping.c:2 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h
// CHECK: {{.*}}{{/|\\}}print-unit-roundtrip-remapping.c:2 | {{.*}}{{/|\\}}Inputs{{/|\\}}sys{{/|\\}}syshead.h
// CHECK-NOT: SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h:1 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}head.h
// CHECK: {{.*}}{{/|\\}}Inputs{{/|\\}}print-unit.h:1 | {{.*}}{{/|\\}}Inputs{{/|\\}}head.h
// CHECK-NOT: SRC_ROOT{{/|\\}}Inputs{{/|\\}}print-unit.h:2 | SRC_ROOT{{/|\\}}Inputs{{/|\\}}using-overlay.h
// CHECK: {{.*}}{{/|\\}}Inputs{{/|\\}}print-unit.h:2 | {{.*}}{{/|\\}}Inputs{{/|\\}}using-overlay.h
// CHECK: INCLUDE END (4)
