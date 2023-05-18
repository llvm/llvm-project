// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -ffreestanding
// expected-no-diagnostics

#import <stdint.h>  // no warning on #import in objc mode.

