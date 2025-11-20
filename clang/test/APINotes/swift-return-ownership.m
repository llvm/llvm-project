// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/SwiftReturnOwnershipForObjC.pcm | FileCheck %s
#import <SwiftReturnOwnershipForObjC.h>

// CHECK: @interface MethodTest
// CHECK: - (struct RefCountedType *)getUnowned __attribute__((swift_attr("returns_unretained")));
// CHECK: - (struct RefCountedType *)getOwned __attribute__((swift_attr("returns_retained")));
// CHECK: @end
// CHECK: __attribute__((swift_attr("returns_unretained"))) struct RefCountedType *getObjCUnowned(void);
// CHECK: __attribute__((swift_attr("returns_retained"))) struct RefCountedType *getObjCOwned(void);
