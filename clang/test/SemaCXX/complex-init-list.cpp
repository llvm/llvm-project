// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic
// expected-no-diagnostics

// This file tests the clang extension which allows initializing the components
// of a complex number individually using an initialization list. Basically,
// if you have an explicit init list for a complex number that contains two
// initializers, this extension kicks in to turn it into component-wise
// initialization.
// 
// See also the testcase for the C version of this extension in
// test/Sema/complex-init-list.c.

// Basic testcase
// (No pedantic warning is necessary because _Complex is not part of C++.)
_Complex float valid1 = { 1.0f, 2.0f };
