// RUN: %clang_cc1 -verify -Wno-objc-root-class %s

@interface FF
- (void) Meth;
@end

@protocol P
@end

@interface INTF<P>
- (void)IMeth;
@end

@implementation INTF
- (void)IMeth {INTF<P> *pi;  [pi Meth]; } // expected-warning {{method '-Meth' not found (return type defaults to 'id')}}
@end
