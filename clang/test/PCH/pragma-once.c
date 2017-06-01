// Test this without pch.
// RUN: %clang_cc1 -include %S/Inputs/pragma-once.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/Inputs/pragma-once.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s

// expected-no-diagnostics

// Including "pragma-once.h" twice, to verify the 'once' aspect is honored.
#include "Inputs/pragma-once.h"
#include "Inputs/pragma-once.h"
int foo(void) { return 0; }
