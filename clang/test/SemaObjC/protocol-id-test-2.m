// RUN: %clang_cc1 -verify -Wno-objc-root-class %s

@protocol P
@end

@interface INTF<P>
- (void)IMeth;
@end

@implementation INTF
- (void)IMeth { [(id<P>)self Meth]; }  // expected-warning {{method '-Meth' not found (return type defaults to 'id')}}
@end
