// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fobjc-runtime=gnustep-2.2 -emit-llvm -o - %s | FileCheck %s

@protocol BaseProtocol
@end

@protocol ExtendedProtocol
@end

@interface TestClass <BaseProtocol>

-(void) Meth;
@end

@interface TestClass () <BaseProtocol, ExtendedProtocol>
@end

@implementation TestClass
@end

// Check that we emit metadata for both protocols
// CHECK: @._OBJC_PROTOCOL_ExtendedProtocol = global
// CHECK: @._OBJC_PROTOCOL_BaseProtocol = global

// Check that we deduplicate the protocol list
// CHECK: @.objc_protocol_list{{\.[0-9]*}} = internal global { ptr, i64, [2 x ptr] }
