// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --product-name=GlobalRecord -triple arm64-apple-macosx \
// RUN: %t/input1.h -verify -o - | FileCheck %s

//--- input1.h
int num;
#include "input2.h"

//--- input2.h
// Ensure that these symbols are not emitted in the Symbol Graph.
#define HELLO 1
char not_emitted;
void foo(int);
struct Foo { int a; };

// CHECK-NOT: input2.h

// expected-no-diagnostics
