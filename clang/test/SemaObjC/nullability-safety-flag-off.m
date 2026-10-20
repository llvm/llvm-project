// With flow-sensitive nullability OFF (the default), ObjC method results and
// parameters in assume_nonnull regions must still get the context-sensitive
// 'nonnull' form, not the _Nonnull keyword form (upstream behavior; affects
// printed types, code completion, and qualifier-merge diagnostics).
//
// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -fblocks -ast-print %s | FileCheck %s

// expected-no-diagnostics

__attribute__((objc_root_class))
@interface A
@end

#pragma clang assume_nonnull begin
@interface A (Pragma)
- (A *)method1:(A *)ptr;
@end
#pragma clang assume_nonnull end

// CHECK: - (nonnull A *)method1:(nonnull A *)ptr;
