// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated-declarations -verify %s

@interface INTF1
@required  // expected-error {{directive may only be specified in protocols only}}
- (int) FooBar;
- (int) FooBar1;
- (int) FooBar2;
@optional  // expected-error {{directive may only be specified in protocols only}}
+ (int) C;

- (int)I;
@end

@protocol p1,p2,p3;

@protocol p1;

@protocol PROTO1
@required 
- (int) FooBar;
@optional
- (void) MyMethod1;
+ (int) S;
@end


@protocol PROTO2<p1>
@end

@protocol p1 @end

@protocol PROTO<p1>     // expected-note {{previous definition is here}}
@end

@protocol PROTO<p1>	// expected-warning {{duplicate protocol definition of 'PROTO'}}
@end

@protocol PROTO3<p1, p1>
@end

@protocol p2 <p1>
@end

@protocol PROTO4 <p1, p2, PROTO, PROTO3, p3> 
@end

@protocol XX;
@protocol YY <XX>  // Use of declaration of XX here should not cause a warning.
- zz;
@end


// Detect circular dependencies.
@protocol B;
@protocol C < B > // expected-note{{previous definition is here}}
@end
@protocol A < C > 
@end
@protocol B < A > // expected-error{{protocol has circular dependency}}
@end

@protocol P
- (int)test:(int)param, ..; // expected-error{{type specifier missing}} \
                      // expected-error{{expected ';' after method prototype}}
@end
