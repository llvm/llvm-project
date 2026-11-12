// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@protocol P1
- (void) P1proto;  // expected-note {{method 'P1proto' declared here}}
+ (void) ClsP1Proto;    // expected-note {{method 'ClsP1Proto' declared here}}
- (void) DefP1proto;
@end
@protocol P2
- (void) P2proto;   // expected-note {{method 'P2proto' declared here}}
+ (void) ClsP2Proto;  // expected-note {{method 'ClsP2Proto' declared here}}
@end

@protocol P3<P2>
- (void) P3proto;   // expected-note {{method 'P3proto' declared here}}
+ (void) ClsP3Proto;   // expected-note {{method 'ClsP3Proto' declared here}}
+ (void) DefClsP3Proto;
@end

@protocol PROTO<P1, P3>
- (void) meth;		  // expected-note {{method 'meth' declared here}}
- (void) meth : (int) arg1;   // expected-note {{method 'meth:' declared here}}
+ (void) cls_meth : (int) arg1;   // expected-note {{method 'cls_meth:' declared here}}
@end

@interface INTF <PROTO>
@end

@implementation INTF // expected-warning 9 {{in protocol '}}
- (void) DefP1proto{}
+ (void) DefClsP3Proto{}
@end
