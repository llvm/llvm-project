// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.UncountedLambdaCapturesChecker %s 2>&1 | FileCheck %s --strict-whitespace
#include "mock-types.h"

void raw_ptr() {
  RefCountable* ref_countable = nullptr;
  auto foo1 = [ref_countable](){};
  // CHECK: warning: Captured raw-pointer 'ref_countable' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
  // CHECK-NEXT:{{^}}  auto foo1 = [ref_countable](){};
  // CHECK-NEXT:{{^}}               ^
  auto foo2 = [&ref_countable](){};
  // CHECK: warning: Captured raw-pointer 'ref_countable' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
  auto foo3 = [&](){ ref_countable = nullptr; };
  // CHECK: warning: Implicitly captured raw-pointer 'ref_countable' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
  // CHECK-NEXT:{{^}}  auto foo3 = [&](){ ref_countable = nullptr; };
  // CHECK-NEXT:{{^}}                     ^
  auto foo4 = [=](){ (void) ref_countable; };
  // CHECK: warning: Implicitly captured raw-pointer 'ref_countable' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
}

void references() {
  RefCountable automatic;
  RefCountable& ref_countable_ref = automatic;

  auto foo1 = [ref_countable_ref](){};
  // CHECK: warning: Captured reference 'ref_countable_ref' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
  auto foo2 = [&ref_countable_ref](){};
  // CHECK: warning: Captured reference 'ref_countable_ref' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
  auto foo3 = [&](){ (void) ref_countable_ref; };
  // CHECK: warning: Implicitly captured reference 'ref_countable_ref' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
  auto foo4 = [=](){ (void) ref_countable_ref; };
  // CHECK: warning: Implicitly captured reference 'ref_countable_ref' to uncounted type is unsafe [webkit.UncountedLambdaCapturesChecker]
}

void quiet() {
// This code is not expected to trigger any warnings.
  {
    RefCountable automatic;
    RefCountable &ref_countable_ref = automatic;
  }

  auto foo3 = [&]() {};
  auto foo4 = [=]() {};
  RefCountable *ref_countable = nullptr;
}
