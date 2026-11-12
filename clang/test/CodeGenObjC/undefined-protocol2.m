// RUN: %clang_cc1 -triple x86_64-apple-macosx -emit-llvm %s -o - | FileCheck %s

// Test that we produce a declaration for the protocol. It must be matched
// by a definition in another TU, so external is the correct linkage
// (not extern_weak).
// CHECK: @"_OBJC_PROTOCOL_$_p1" = external global

@interface NSObject
@end

@protocol p1;

@interface I1 : NSObject <p1>
@end

@implementation I1
@end
