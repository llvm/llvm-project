// UNSUPPORTED: -zos, -aix
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -emit-llvm -o %t/test-compatible-extensions.ll -fobjc-runtime=macosx-10.9 -F%t/Frameworks %t/test-compatible-extensions.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -fmodule-name=InterfaceAndExtension
// RUN: FileCheck --input-file=%t/test-compatible-extensions.ll %t/test-compatible-extensions.m

// RUN: %clang_cc1 -emit-llvm -o %t/test-access-extension-ivar.ll -fobjc-runtime=macosx-10.9 -F%t/Frameworks %t/test-access-extension-ivar.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: FileCheck --input-file=%t/test-access-extension-ivar.ll %t/test-access-extension-ivar.m

// RUN: %clang_cc1 -emit-llvm -o %t/test-synthesized-ivar.ll -fobjc-runtime=macosx-10.9 -F%t/Frameworks %t/test-synthesized-ivar.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: FileCheck --input-file=%t/test-synthesized-ivar.ll %t/test-synthesized-ivar.m
// RUN: %clang_cc1 -emit-llvm -o %t/test-synthesized-ivar-extension.ll -fobjc-runtime=macosx-10.9 -F%t/Frameworks %t/test-synthesized-ivar.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache \
// RUN:            -DIMPORT_EXTENSION=1
// RUN: FileCheck --input-file=%t/test-synthesized-ivar-extension.ll %t/test-synthesized-ivar.m

// Test various scenarios where we can end up with the same-name ivars coming from multiple modules.
// The goal is to avoid duplicate metadata for ivars because it can lead to miscompilations
// with a wrong ivar offset.
//
// See specific .m tests for the details of various scenarios.

//--- Frameworks/InterfaceAndExtension.framework/Headers/Interface.h
@interface NSObject @end
@interface ObjCInterface : NSObject
@end

//--- Frameworks/InterfaceAndExtension.framework/Headers/Extension.h
#import <InterfaceAndExtension/Interface.h>
@interface ObjCInterface() {
  float ivarInExtension;
  int bitfieldIvarInExtension: 3;
}
@end

//--- Frameworks/InterfaceAndExtension.framework/Headers/InterfaceAndExtension.h
#import <InterfaceAndExtension/Interface.h>
#import <InterfaceAndExtension/Extension.h>

//--- Frameworks/InterfaceAndExtension.framework/Modules/module.modulemap
framework module InterfaceAndExtension {
  umbrella header "InterfaceAndExtension.h"
  export *
  module * { export * }
}

//--- Frameworks/Redirecting.framework/Headers/Redirecting.h
#import <InterfaceAndExtension/InterfaceAndExtension.h>

//--- Frameworks/Redirecting.framework/Modules/module.modulemap
framework module Redirecting {
  header "Redirecting.h"
  export *
}

//--- test-compatible-extensions.m
// Test adding through deserialization an extension with already declared ivars.

// First create `ObjCInterface()` extension by parsing corresponding code.
#import <InterfaceAndExtension/InterfaceAndExtension.h>
// Now add the same extension through deserialization from the imported module.
#import <Redirecting/Redirecting.h>
@implementation ObjCInterface {
  int ivarInImplementation;
}
@end
// CHECK: @"_OBJC_$_INSTANCE_VARIABLES_ObjCInterface"
// CHECK-SAME: [3 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_ObjCInterface.ivarInExtension", {{.*}} }, %struct._ivar_t { ptr @"OBJC_IVAR_$_ObjCInterface.bitfieldIvarInExtension", {{.*}} }, %struct._ivar_t { ptr @"OBJC_IVAR_$_ObjCInterface.ivarInImplementation", {{.*}} }]


//--- Frameworks/WithInlineIvar.framework/Headers/WithInlineIvar.h
#import <InterfaceAndExtension/InterfaceAndExtension.h>
@interface ObjCInterface() {
@public
  int accessedIvar;
}
@end
static inline void inlinedIvarAccessor(ObjCInterface *obj) {
  obj->accessedIvar = 0;
}

//--- Frameworks/WithInlineIvar.framework/Modules/module.modulemap
framework module WithInlineIvar {
  header "WithInlineIvar.h"
  export *
}

//--- test-access-extension-ivar.m
// Test accessing ivars from extensions.
#import <InterfaceAndExtension/InterfaceAndExtension.h>
@interface ObjCInterface() {
@public
  int accessedIvar;
}
@end
#import <WithInlineIvar/WithInlineIvar.h>
@implementation ObjCInterface
- (void)test {
  inlinedIvarAccessor(self);
  ivarInExtension = 0;
}
@end
// CHECK: @"_OBJC_$_INSTANCE_VARIABLES_ObjCInterface"
// CHECK-SAME: [3 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_ObjCInterface.accessedIvar", {{.*}} }, %struct._ivar_t { ptr @"OBJC_IVAR_$_ObjCInterface.ivarInExtension", {{.*}} }, %struct._ivar_t { ptr @"OBJC_IVAR_$_ObjCInterface.bitfieldIvarInExtension", {{.*}} }]


//--- Frameworks/WithProperty.framework/Headers/WithProperty.h
@interface NSObject @end
@interface WithProperty: NSObject
@property (assign) int propertyName;
@end

//--- Frameworks/WithProperty.framework/Modules/module.modulemap
framework module WithProperty {
  header "WithProperty.h"
  export *
}

//--- Frameworks/BackingIvarInExtension.framework/Headers/BackingIvarInExtension.h
#import <WithProperty/WithProperty.h>
@interface WithProperty() {
  int propertyBackingIvar;
}
@end

//--- Frameworks/BackingIvarInExtension.framework/Modules/module.modulemap
framework module BackingIvarInExtension {
  header "BackingIvarInExtension.h"
  export *
}

//--- test-synthesized-ivar.m
// Test when an ivar is both synthesized and when declared in an extension.
// Behavior with and without extension should be the same.
#import <WithProperty/WithProperty.h>
#ifdef IMPORT_EXTENSION
#import <BackingIvarInExtension/BackingIvarInExtension.h>
#endif
@implementation WithProperty
@synthesize propertyName = propertyBackingIvar;
@end
// CHECK: @"_OBJC_$_INSTANCE_VARIABLES_WithProperty"
// CHECK-SAME: [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_WithProperty.propertyBackingIvar", {{.*}} }]
