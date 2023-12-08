// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol MyProto1 
@optional
- (void) FOO;
@optional
- (void) FOO1;
@required 
- (void) REQ;
@optional
@end

@protocol  MyProto2 <MyProto1>
- (void) FOO2;
@optional
- (void) FOO3;
@end
