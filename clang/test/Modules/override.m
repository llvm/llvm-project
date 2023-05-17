// UNSUPPORTED: target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -I%t/include %t/test.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -fmodule-name=CheckOverride

// Test that if we have the same method in a different module, it's not an
// override as it is the same method and it has the same DeclContext but a
// different object in the memory.


//--- include/CheckOverride.h
@interface NSObject
@end

@interface CheckOverrideInterfaceOnly: NSObject
- (void)potentialOverrideInterfaceOnly;
@end

@interface CheckOverrideCategoryOnly: NSObject
@end
@interface CheckOverrideCategoryOnly(CategoryOnly)
- (void)potentialOverrideCategoryOnly;
@end

@interface CheckOverrideImplementationOfInterface: NSObject
- (void)potentialOverrideImplementationOfInterface;
@end

@interface CheckOverrideImplementationOfCategory: NSObject
@end
@interface CheckOverrideImplementationOfCategory(CategoryImpl)
- (void)potentialOverrideImplementationOfCategory;
@end

//--- include/Redirect.h
// Ensure CheckOverride is imported as the module despite all `-fmodule-name` flags.
#import <CheckOverride.h>

//--- include/module.modulemap
module CheckOverride {
  header "CheckOverride.h"
}
module Redirect {
  header "Redirect.h"
  export *
}

//--- test.m
#import <CheckOverride.h>
#import <Redirect.h>

@implementation CheckOverrideImplementationOfInterface
- (void)potentialOverrideImplementationOfInterface {}
@end

@implementation CheckOverrideImplementationOfCategory
- (void)potentialOverrideImplementationOfCategory {}
@end

void triggerOverrideCheck(CheckOverrideInterfaceOnly *intfOnly,
                          CheckOverrideCategoryOnly *catOnly,
                          CheckOverrideImplementationOfInterface *intfImpl,
                          CheckOverrideImplementationOfCategory *catImpl) {
  [intfOnly potentialOverrideInterfaceOnly];
  [catOnly potentialOverrideCategoryOnly];
  [intfImpl potentialOverrideImplementationOfInterface];
  [catImpl potentialOverrideImplementationOfCategory];
}
