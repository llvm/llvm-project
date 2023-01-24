// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang -cc1depscan -o %t/inline.rsp -fdepscan=inline -fdepscan-include-tree -cc1-args -cc1 -triple x86_64-apple-macos11.0 \
// RUN:     -emit-obj %t/t.c -o %t/t.o -dwarf-ext-refs -fmodule-format=obj \
// RUN:     -I %t/includes -isysroot %S/Inputs/SDK -fcas-path %t/cas -DSOME_MACRO -dependency-file %t/inline.d -MT deps

// RUN: FileCheck %s -input-file %t/inline.rsp -DPREFIX=%t
// RUN: FileCheck %s -input-file %t/inline.rsp -DPREFIX=%t -check-prefix=SHOULD

// RUN: %clang @%t/inline.rsp

// CHECK: "-fcas-path" "[[PREFIX]]/cas"
// CHECK: "-fcas-include-tree"
// CHECK: "-isysroot"
// SHOULD-NOT: "-fcas-fs"
// SHOULD-NOT: "-fcas-fs-working-directory"
// SHOULD-NOT: "-I"
// SHOULD-NOT: "[[PREFIX]]/t.c"
// SHOULD-NOT: "-D"
// SHOULD-NOT: "-dwarf-ext-refs"
// SHOULD-NOT: "-fmodule-format=obj"

// RUN: FileCheck %s -input-file %t/inline.d -check-prefix=DEPS -DPREFIX=%t

// DEPS: deps:
// DEPS: [[PREFIX]]/t.c
// DEPS: [[PREFIX]]/includes/t.h

//--- t.c
#include "t.h"

int test(struct S *s) {
  return s->x;
}

//--- includes/t.h
struct S {
  int x;
};
