// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-mismatch-in-extension.m
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-mismatch-in-extension.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-mismatch-in-ivars-number.m
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-mismatch-in-ivars-number.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-mismatch-in-methods-protocols.m
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-mismatch-in-methods-protocols.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-redecl-in-subclass.m
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-redecl-in-subclass.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-redecl-in-implementation.m
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-10.9 -verify -I%t/include %t/test-redecl-in-implementation.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Same class extensions with the same ivars but from different modules aren't considered
// an error and they are merged together. Test that differences in extensions and/or ivars
// are still reported as errors.

//--- include/Interfaces.h
@interface NSObject @end
@interface ObjCInterface : NSObject
@end
@interface ObjCInterfaceLevel2 : ObjCInterface
@end

@protocol Protocol1 @end
@protocol Protocol2 @end

//--- include/IvarsInExtensions.h
#import <Interfaces.h>
@interface ObjCInterface() {
  int ivarName;
}
@end
@interface ObjCInterfaceLevel2() {
  int bitfieldIvarName: 3;
}
@end

//--- include/IvarsInExtensionsWithMethodsProtocols.h
#import <Interfaces.h>
@interface ObjCInterface() {
  int methodRelatedIvar;
}
- (void)test;
@end
@interface ObjCInterfaceLevel2() <Protocol1> {
  int protocolRelatedIvar;
}
@end

//--- include/IvarInImplementation.h
#import <Interfaces.h>
@implementation ObjCInterface {
  int ivarName;
}
@end

//--- include/module.modulemap
module Interfaces {
  header "Interfaces.h"
  export *
}
module IvarsInExtensions {
  header "IvarsInExtensions.h"
  export *
}
module IvarsInExtensionsWithMethodsProtocols {
  header "IvarsInExtensionsWithMethodsProtocols.h"
  export *
}
module IvarInImplementation {
  header "IvarInImplementation.h"
  export *
}


//--- test-mismatch-in-extension.m
// Different ivars with the same name aren't mergeable and constitute an error.
#import <Interfaces.h>
@interface ObjCInterface() {
  float ivarName; // expected-note {{previous definition is here}}
}
@end
@interface ObjCInterfaceLevel2() {
  int bitfieldIvarName: 5; // expected-note {{previous definition is here}}
}
@end
#import <IvarsInExtensions.h>
// expected-error@IvarsInExtensions.h:* {{instance variable is already declared}}
// expected-error@IvarsInExtensions.h:* {{instance variable is already declared}}
@implementation ObjCInterfaceLevel2
@end


//--- test-mismatch-in-ivars-number.m
// Extensions with different amount of ivars aren't considered to be the same.
#import <Interfaces.h>
@interface ObjCInterface() {
  int ivarName; // expected-note {{previous definition is here}}
  float anotherIvar;
}
@end
#import <IvarsInExtensions.h>
// expected-error@IvarsInExtensions.h:* {{instance variable is already declared}}
@implementation ObjCInterface
@end


//--- test-mismatch-in-methods-protocols.m
// Extensions with different methods or protocols aren't considered to be the same.
#import <Interfaces.h>
@interface ObjCInterface() {
  int methodRelatedIvar; // expected-note {{previous definition is here}}
}
- (void)differentTest;
@end
@interface ObjCInterfaceLevel2() <Protocol2> {
  int protocolRelatedIvar; // expected-note {{previous definition is here}}
}
@end
#import <IvarsInExtensionsWithMethodsProtocols.h>
// expected-error@IvarsInExtensionsWithMethodsProtocols.h:* {{instance variable is already declared}}
// expected-error@IvarsInExtensionsWithMethodsProtocols.h:* {{instance variable is already declared}}
@implementation ObjCInterfaceLevel2
@end


//--- test-redecl-in-subclass.m
// Ivar in superclass extension is not added to a subclass, so the ivar with
// the same name in subclass extension is not considered a redeclaration.
// expected-no-diagnostics
#import <Interfaces.h>
@interface ObjCInterfaceLevel2() {
  float ivarName;
}
@end
#import <IvarsInExtensions.h>
@implementation ObjCInterfaceLevel2
@end


//--- test-redecl-in-implementation.m
// Ivar redeclaration in `@implementation` is always an error and never mergeable.
#import <IvarsInExtensions.h>
@interface ObjCInterface() {
  int triggerExtensionIvarDeserialization;
}
@end
#import <IvarInImplementation.h>
#if __has_feature(modules)
// expected-error@IvarsInExtensions.h:* {{instance variable is already declared}}
// expected-note@IvarInImplementation.h:* {{previous definition is here}}
#else
// expected-error@IvarInImplementation.h:* {{instance variable is already declared}}
// expected-note@IvarsInExtensions.h:* {{previous definition is here}}
#endif
