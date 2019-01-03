// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s

// Regression test.  We weren't emitting definitions for protocols used in
// categories, causing linker errors when the category was the only reference
// to a protocol in a binary.

// CHECK: @._OBJC_PROTOCOL_Y = global 
// CHEKC-SAME: section "__objc_protocols", comdat, align 8


@interface X
{
id isa;
}
@end
@implementation X
@end

@protocol Y @end

@interface X (y) <Y>
@end
@implementation X (y) @end


