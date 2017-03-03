// RUN: %clang_analyze_cc1 %s -analyzer-checker=core,osx.cocoa.RetainCount -fblocks -verify

// This test is checking behavior when a single checker runs only with the core
// checkers, testing that the traversal order in the CFG does not affect the
// reporting of an error.

#import "Inputs/system-header-simulator-objc.h"

void testDoubleRelease(BOOL z) {
  id x = [[NSObject alloc] init];
  if (z) {
    [x release];
  } else {
    ;
  }
  [x release]; // expected-warning {{Reference-counted object is used after it is released}}
}

void testDoubleRelease2(BOOL z) {
  id x = [[NSObject alloc] init];
  if (z) {
    ;
  } else {
    [x release];
  }
  [x release]; // expected-warning {{Reference-counted object is used after it is released}}
}
