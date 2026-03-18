// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -emit-llvm -fobjc-runtime=gnustep-2.2 -o - %s | FileCheck %s

@class NSString;

@protocol AProtocol
- (void) meth;
@end

@interface AClass <AProtocol>
@end

@implementation AClass
- (void) meth {}
@end

// Make sure that all public symbols are mangled correctly. All exported symbols
// must be valid Javascript identifiers in Emscripten.
// CHECK: $"$_OBJC_PROTOCOL_AProtocol" = comdat any
// CHECK: @"$_OBJC_METACLASS_AClass"
// CHECK: @"$_OBJC_PROTOCOL_AProtocol"
// CHECK: @"$_OBJC_CLASS_AClass"
// CHECK: @"$_OBJC_REF_CLASS_AClass"
// CHECK: @"$_OBJC_INIT_CLASS_AClass"
