// RUN: %clang_cc1 -fobjc-arc -emit-pch -o %t.pch %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t.pch | FileCheck %s

// All three non-null types below should share a single AttributedType node and
// serialize as exactly one TYPE_ATTRIBUTED record.

@class NSString;

#pragma clang assume_nonnull begin
@interface T
- (NSString *)a;
- (NSString *)b;
- (NSString *)c;
@end
#pragma clang assume_nonnull end

// CHECK-COUNT-1: <TYPE_ATTRIBUTED
// CHECK-NOT:     <TYPE_ATTRIBUTED
