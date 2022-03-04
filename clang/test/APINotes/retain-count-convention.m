// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules  -fdisable-module-hash -fsyntax-only -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/SimpleKit.pcm | FileCheck %s
// RUN: %clang_cc1 -ast-dump -ast-dump-filter 'DUMP' %t/ModulesCache/SimpleKit.pcm | FileCheck -check-prefix CHECK-DUMP %s

#import <SimpleKit/SimpleKit.h>

// CHECK: void *getCFOwnedToUnowned(void) __attribute__((cf_returns_not_retained));
// CHECK: void *getCFUnownedToOwned(void) __attribute__((cf_returns_retained));
// CHECK: void *getCFOwnedToNone(void) __attribute__((cf_unknown_transfer));
// CHECK: id getObjCOwnedToUnowned(void) __attribute__((ns_returns_not_retained));
// CHECK: id getObjCUnownedToOwned(void) __attribute__((ns_returns_retained));
// CHECK: int indirectGetCFOwnedToUnowned(void * _Nullable *out __attribute__((cf_returns_not_retained)));
// CHECK: int indirectGetCFUnownedToOwned(void * _Nullable *out __attribute__((cf_returns_retained)));
// CHECK: int indirectGetCFOwnedToNone(void * _Nullable *out);
// CHECK: int indirectGetCFNoneToOwned(void **out __attribute__((cf_returns_not_retained)));

// CHECK-LABEL: @interface MethodTest
// CHECK: - (id)getOwnedToUnowned __attribute__((ns_returns_not_retained));
// CHECK: - (id)getUnownedToOwned __attribute__((ns_returns_retained));
// CHECK: @end

// CHECK-DUMP-LABEL: Dumping getCFAuditedToUnowned_DUMP:
// CHECK-DUMP-NEXT: FunctionDecl
// CHECK-DUMP-NEXT: CFReturnsNotRetainedAttr
// CHECK-DUMP-NEXT: CFAuditedTransferAttr
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping getCFAuditedToOwned_DUMP:
// CHECK-DUMP-NEXT: FunctionDecl
// CHECK-DUMP-NEXT: CFReturnsRetainedAttr
// CHECK-DUMP-NEXT: CFAuditedTransferAttr
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping getCFAuditedToNone_DUMP:
// CHECK-DUMP-NEXT: FunctionDecl
// CHECK-DUMP-NEXT: CFUnknownTransferAttr
// CHECK-DUMP-NOT: Attr
