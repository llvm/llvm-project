// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wno-objc-root-class -I %t %t/repro.m -verify

//--- header.h
#define DECLARE_MACRO_METHOD() \
  - (void)macroMethod {}

//--- repro.m
#include "header.h"
/**
 * A multiline doc comment whose byte offset range
 * straddles the macro method offset in header.h.
 */
@interface Foo
@end

@implementation Foo
- (void)normalMethod {}
DECLARE_MACRO_METHOD()
@end
// expected-no-diagnostics
