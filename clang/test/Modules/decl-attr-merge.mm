// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// macOS: single-platform availability worked even before the fix.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.dir/mcache -triple x86_64-apple-macosx10.11.0 \
// RUN:   -I%t.dir/headers %t.dir/main-macos.m -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MACOS
// iOS: the @interface carries two AvailabilityAttrs (macOS + iOS).
// Without the fix, mergeInheritableAttributes used getAttr<AvailabilityAttr>()
// which only copied the first (macOS); the iOS attr was lost on the @class
// redeclaration, causing isWeakImported() to return false (strong linkage).
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.dir/icache -triple arm64-apple-ios12.0 \
// RUN:   -I%t.dir/headers %t.dir/main-ios.m -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-IOS

//--- headers/a.h

__attribute__((availability(macos,introduced=10.16)))
__attribute__((availability(ios,introduced=14.0)))
@interface INIntent
- (instancetype)self;
@end

//--- headers/b.h

@class INIntent;

//--- headers/module.modulemap

module A {
  header "a.h"
}

module B {
  header "b.h"
}

//--- main-macos.m

#import <a.h>
#import <b.h> // NOTE: Non attributed decl imported after one with attrs.

void F(id);

int main() {
  if (@available(macOS 11.0, *))
    F([INIntent self]);
}

// CHECK-MACOS: @"OBJC_CLASS_$_INIntent" = extern_weak

//--- main-ios.m

#import <a.h>
#import <b.h>

@implementation INIntent (Testing)
@end

// CHECK-IOS: @"OBJC_CLASS_$_INIntent" = extern_weak
