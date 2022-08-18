// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang -cc1depscan -o %t/inline.rsp -fdepscan=inline -fdepscan-include-tree -cc1-args -cc1 -triple x86_64-apple-macos11.0 \
// RUN:     -fsyntax-only %t/t.c -I %t/includes -isysroot %S/Inputs/SDK -fcas-path %t/cas -DSOME_MACRO -dependency-file %t/inline.d -MT deps
// RUN: %clang -cc1depscan -o %t/daemon.rsp -fdepscan=daemon -fdepscan-include-tree -cc1-args -cc1 -triple x86_64-apple-macos11.0 \
// RUN:     -fsyntax-only %t/t.c -I %t/includes -isysroot %S/Inputs/SDK -fcas-path %t/cas -DSOME_MACRO -dependency-file %t/daemon.d -MT deps

// RUN: diff -u %t/inline.rsp %t/daemon.rsp
// RUN: FileCheck %s -input-file %t/inline.rsp -DPREFIX=%t
// RUN: FileCheck %s -input-file %t/inline.rsp -DPREFIX=%t -check-prefix=SHOULD

// RUN: %clang @%t/inline.rsp

// CHECK: "-fcas-path" "[[PREFIX]]/cas"
// CHECK: "-fcas-include-tree"
// SHOULD-NOT: "-fcas-fs"
// SHOULD-NOT: "-fcas-fs-working-directory"
// SHOULD-NOT: "-isysroot"
// SHOULD-NOT: "-I"
// SHOULD-NOT: "[[PREFIX]]/t.c"
// SHOULD-NOT: "-D"

// RUN: diff -u %t/inline.d %t/daemon.d
// RUN: FileCheck %s -input-file %t/inline.d -check-prefix=DEPS -DPREFIX=%t

// DEPS: deps:
// DEPS: [[PREFIX]]/t.c
// DEPS: [[PREFIX]]/includes/t.h

int test() { return 0; }

//--- t.c
#include "t.h"

int test(struct S *s) {
  return s->x;
}

//--- includes/t.h
struct S {
  int x;
};
