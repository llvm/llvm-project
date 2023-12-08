// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -fblocks -fobjc-arc -w -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true %s -verify

#include "Inputs/system-header-simulator-objc.h"

NSNumber* generateNumber();

// expected-no-diagnostics
int test_initialization_in_ifstmt() { // Don't warn on initialization in guard.
  if (NSNumber* number = generateNumber()) { // no-warning
    return 0;
  }
  return 1;
}
