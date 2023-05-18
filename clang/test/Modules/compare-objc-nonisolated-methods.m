// RUN: rm -rf %t
// RUN: split-file %s %t

// Test that different values of `ObjCMethodDecl::isOverriding` in different modules
// is not an error because it depends on the surrounding code and not on the method itself.
// RUN: %clang_cc1 -fsyntax-only -verify -I%t/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -fmodule-name=Override %t/test-overriding.m

// Test that different values of `ObjCMethodDecl::isPropertyAccessor` in different modules
// is not an error because it depends on the surrounding code and not on the method itself.
// RUN: %clang_cc1 -fsyntax-only -verify -I%t/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -fmodule-name=PropertyAccessor %t/test-property_accessor.m

//--- include/Common.h
@interface NSObject
@end

//--- include/Indirection.h
#import <Override.h>
#import <PropertyAccessor.h>

//--- include/module.modulemap
module Common {
  header "Common.h"
  export *
}
module Indirection {
  header "Indirection.h"
  export *
}
module Override {
  header "Override.h"
  export *
}
module PropertyAccessor {
  header "PropertyAccessor.h"
  export *
}

//--- include/Override.h
#import <Common.h>
@interface SubClass: NSObject
- (void)potentialOverride;
@end

//--- Override_Internal.h
#import <Common.h>
@interface NSObject(InternalCategory)
- (void)potentialOverride;
@end

//--- test-overriding.m
//expected-no-diagnostics
// Get non-modular version of `SubClass`, so that `-[SubClass potentialOverride]`
// is an override of a method in `InternalCategory`.
#import "Override_Internal.h"
#import <Override.h>

// Get modular version of `SubClass` where `-[SubClass potentialOverride]` is
// not an override because module "Override" doesn't know about Override_Internal.h.
#import <Indirection.h>

void triggerOverrideCheck(SubClass *sc) {
  [sc potentialOverride];
}

//--- include/PropertyAccessor.h
#import <Common.h>
@interface PropertySubClass: NSObject
- (int)potentialProperty;
- (void)setPotentialProperty:(int)p;
@end

//--- PropertyAccessor_Internal.h
#import <PropertyAccessor.h>
@interface PropertySubClass()
@property int potentialProperty;
@end

//--- test-property_accessor.m
//expected-no-diagnostics
// Get a version of `PropertySubClass` where `-[PropertySubClass potentialProperty]`
// is a property accessor.
#import "PropertyAccessor_Internal.h"

// Get a version of `PropertySubClass` where `-[PropertySubClass potentialProperty]`
// is not a property accessor because module "PropertyAccessor" doesn't know about PropertyAccessor_Internal.h.
#import <Indirection.h>

void triggerPropertyAccessorCheck(PropertySubClass *x) {
  int tmp = [x potentialProperty];
  [x setPotentialProperty: tmp];
}
