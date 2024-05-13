// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -Wobjc-multiple-method-names -fsyntax-only -fmodules-cache-path=%t/modules.cache -fmodules -fimplicit-module-maps -F %t/Frameworks %t/test.m -verify

// Verify we are handling methods from transitive modules, not just from immediate ones.

//--- Frameworks/Indirect.framework/Headers/Indirect.h
@interface NSObject
@end

@interface Indirect : NSObject
- (int)method;
@end

//--- Frameworks/Indirect.framework/Modules/module.modulemap
framework module Indirect {
  header "Indirect.h"
  export *
}

//--- Frameworks/Immediate.framework/Headers/Immediate.h
#import <Indirect/Indirect.h>
@interface Immediate : NSObject
- (void)method;
@end

//--- Frameworks/Immediate.framework/Modules/module.modulemap
framework module Immediate {
  header "Immediate.h"
  export *
}

//--- test.m
#import <Immediate/Immediate.h>

void test(id obj) {
  [obj method];  // expected-warning{{multiple methods named 'method' found}}
  // expected-note@Frameworks/Indirect.framework/Headers/Indirect.h:5{{using}}
  // expected-note@Frameworks/Immediate.framework/Headers/Immediate.h:3{{also found}}
}
