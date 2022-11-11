// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/A.framework/Modules/module.modulemap
framework module A {
  umbrella header "A.h"
  exclude header "Excluded.h"

  module Excluded {
    header "Excluded.h"
  }
}
//--- frameworks/A.framework/Headers/A.h
#import <A/Sub.h>
//--- frameworks/A.framework/Headers/Sub.h
//--- frameworks/A.framework/Headers/Excluded.h
#import <A/Sub.h>
@interface I
@end

//--- tu.m
#import <A/Excluded.h>

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -iframework %t/frameworks -fsyntax-only %t/tu.m -verify
// expected-no-diagnostics
