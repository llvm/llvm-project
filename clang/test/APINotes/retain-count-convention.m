// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules  -fdisable-module-hash -fsyntax-only -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/SimpleKit.pcm | FileCheck %s

#import <SimpleKit/SimpleKit.h>

// CHECK: void *getCFOwnedToUnowned() __attribute__((cf_returns_not_retained));
// CHECK: void *getCFUnownedToOwned() __attribute__((cf_returns_retained));
// CHECK: void *getCFOwnedToNone();
// CHECK: id getObjCOwnedToUnowned() __attribute__((ns_returns_not_retained));
// CHECK: id getObjCUnownedToOwned() __attribute__((ns_returns_retained));
// CHECK: int indirectGetCFOwnedToUnowned(void * _Nullable *out __attribute__((cf_returns_not_retained)));
// CHECK: int indirectGetCFUnownedToOwned(void * _Nullable *out __attribute__((cf_returns_retained)));
// CHECK: int indirectGetCFOwnedToNone(void * _Nullable *out);
// CHECK: int indirectGetCFNoneToOwned(void **out __attribute__((cf_returns_not_retained)));

// CHECK-LABEL: @interface MethodTest
// CHECK: - (id)getOwnedToUnowned __attribute__((ns_returns_not_retained));
// CHECK: - (id)getUnownedToOwned __attribute__((ns_returns_retained));
// CHECK: @end
