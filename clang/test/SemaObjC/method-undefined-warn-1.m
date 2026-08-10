// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface INTF
- (void) meth;
- (void) meth : (int) arg1;
- (int) int_meth;  // expected-note {{method 'int_meth' declared here}}
+ (int) cls_meth;  // expected-note {{method 'cls_meth' declared here}}
+ (void) cls_meth1 : (int) arg1;  // expected-note {{method 'cls_meth1:' declared here}}
@end

@implementation INTF // expected-warning {{method definition for 'int_meth' not found}} \
                     // expected-warning {{method definition for 'cls_meth' not found}} \
                     // expected-warning {{method definition for 'cls_meth1:' not found}}
- (void) meth {}
- (void) meth : (int) arg2{}
- (void) cls_meth1 : (int) arg2{}
@end

@interface INTF1
- (void) meth;
- (void) meth : (int) arg1;
- (int)  int_meth; // expected-note {{method 'int_meth' declared here}}
+ (int) cls_meth;  // expected-note {{method 'cls_meth' declared here}}
+ (void) cls_meth1 : (int) arg1;  // expected-note {{method 'cls_meth1:' declared here}}
@end

@implementation INTF1 // expected-warning {{method definition for 'int_meth' not found}} \
                      // expected-warning {{method definition for 'cls_meth' not found}} \
                      // expected-warning {{method definition for 'cls_meth1:' not found}}
- (void) meth {}
- (void) meth : (int) arg2{}
- (void) cls_meth1 : (int) arg2{}
@end

@interface INTF2
- (void) meth;
- (void) meth : (int) arg1;
- (void) cls_meth1 : (int) arg1; 
@end

@implementation INTF2
- (void) meth {}
- (void) meth : (int) arg2{}
- (void) cls_meth1 : (int) arg2{}
@end

@interface Root @end

@interface Foo : Root @end

@implementation Foo

- (void)someFunction { return; }

+ (void)anotherFunction {
    [self someFunction]; // expected-warning {{method '+someFunction' not found (return type defaults to 'id')}}
}
@end
