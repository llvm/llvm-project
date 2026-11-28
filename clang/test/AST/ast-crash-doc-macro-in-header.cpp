// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -I %t %t/repro.cpp -verify

//--- header.h
#define DECL_B b
#define MY_INT int

//--- repro.cpp
#include "header.h"
/**
 * A multiline doc comment whose byte offset range
 * straddles the macro offset in header.h.
 */
void normalFunc();

MY_INT a, DECL_B;
// expected-no-diagnostics
