// Makes sure it doesn't crash.

// XFAIL: linux

// RUN: rm -rf %t
// RUN: not %clang_cc1 %s -index-store-path %t/idx -std=c++14
// RUN: c-index-test core -print-record %t/idx | FileCheck %s

namespace rdar32474406 {
void foo();
typedef void (*Func_t)();
// CHECK: [[@LINE+4]]:1 | type-alias/C | c:record-hash-crash-invalid-name.cpp@N@rdar32474406@T@Func_t | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | c:@N@rdar32474406
// CHECK: [[@LINE+2]]:14 | function/C | c:@N@rdar32474406@F@foo# | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | c:@N@rdar32474406
Func_t[] = { foo }; // invalid decomposition
}
