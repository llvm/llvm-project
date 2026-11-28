// RUN: %clang_cc1 -triple x86_64-apple-macosx -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s

// @encode and method type encodings encode an overflow behavior type as its
// underlying type.

// CHECK: c"i\00"
const char *e = @encode(int __ob_trap);

@interface Foo
- (int __ob_trap)m:(unsigned __ob_wrap)x;
@end

// CHECK: c"i20@0:8I16\00"
@implementation Foo
- (int __ob_trap)m:(unsigned __ob_wrap)x { return 0; }
@end
