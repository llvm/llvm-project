// RUN: %clang_cc1 -emit-llvm -o %t %s
// Test that meta-data for ivar lists with unnamed bitfield are generated.
//
@interface Foo {
@private
    int first;
    int :1;
    int third :1;
    int :1;
    int fifth :1;
}
@end
@implementation Foo 
@end
