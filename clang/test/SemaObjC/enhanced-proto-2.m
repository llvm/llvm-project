// RUN: %clang_cc1 -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@protocol MyProto1 
@optional
- (void) FOO;
@optional
- (void) FOO1;
@optional 
- (void) REQ;
@optional
@end

@interface  MyProto2 <MyProto1>
- (void) FOO2;
- (void) FOO3;
@end

@implementation MyProto2
- (void) FOO2{}
- (void) FOO3{}
@end
