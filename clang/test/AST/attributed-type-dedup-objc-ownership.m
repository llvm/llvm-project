// RUN: %clang_cc1 -fobjc-arc -emit-pch -o %t.pch %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t.pch | FileCheck %s

@class NSString;

__attribute__((objc_ownership(strong))) NSString *a;
__attribute__((objc_ownership(strong))) NSString *b;
__attribute__((objc_ownership(none)))   NSString *c;

// CHECK-COUNT-2: <TYPE_ATTRIBUTED
// CHECK-NOT:     <TYPE_ATTRIBUTED
