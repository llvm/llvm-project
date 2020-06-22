// Test for assertion failure due to objc_externally_retained on a function.

// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-arc -include %s %s

// With PCH
// RUN: %clang_cc1 %s -emit-pch -fobjc-arc -o %t
// RUN: %clang_cc1 -emit-llvm-only -verify %s -fobjc-arc -include-pch %t -debug-info-kind=limited

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
//===----------------------------------------------------------------------===//
// Header

__attribute__((objc_externally_retained)) void doSomething(id someObject);

id sharedObject = 0;

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

void callDoSomething() {
  doSomething(sharedObject);
}

//===----------------------------------------------------------------------===//
#endif
