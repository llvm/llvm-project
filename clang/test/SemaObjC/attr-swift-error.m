// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

@class NSError;

#if __SIZEOF_POINTER__ == 4
typedef unsigned char BOOL;
#else
typedef _Bool BOOL;
#endif

typedef struct __attribute__((__objc_bridge__(NSError))) __CFError *CFErrorRef;

extern int f0(void) __attribute__((__swift_error__));
// expected-error@-1 {{'__swift_error__' attribute takes one argument}}
extern int f1(void) __attribute__((__swift_error__(invalid)));
// expected-warning@-1 {{'__swift_error__' attribute argument not supported: 'invalid'}}
extern int f2(void) __attribute__((__swift_error__(none,zero_result)));
// expected-error@-1 {{use of undeclared identifier 'zero_result'}}

@interface Erroneous
- (BOOL)m0:(NSError **)error __attribute__((__swift_error__(none)));
- (BOOL)m1:(NSError **)error __attribute__((__swift_error__(nonnull_error)));
- (BOOL)m2:(NSError **)error __attribute__((__swift_error__(null_result)));
// expected-error@-1 {{'__swift_error__' attribute with 'null_result' convention can only be applied to a method returning a pointer}}
- (BOOL)m3:(NSError **)error __attribute__((__swift_error__(nonzero_result)));
- (BOOL)m4:(NSError **)error __attribute__((__swift_error__(zero_result)));

- (Undeclared)n0:(NSError **)error __attribute__((__swift_error__(none)));
// expected-error@-1 {{expected a type}}
- (Undeclared)n1:(NSError **)error __attribute__((__swift_error__(nonnull_error)));
// expected-error@-1 {{expected a type}}
- (Undeclared)n2:(NSError **)error __attribute__((__swift_error__(null_result)));
// expected-error@-1 {{expected a type}}
- (Undeclared)n3:(NSError **)error __attribute__((__swift_error__(nonzero_result)));
// expected-error@-1 {{expected a type}}
// FIXME: the follow-on warning should really be suppressed, but apparently
// having an ill-formed return type doesn't mark anything as invalid.
// expected-error@-4 {{can only be applied}}
- (Undeclared)n4:(NSError **)error __attribute__((__swift_error__(zero_result)));
// expected-error@-1 {{expected a type}}
// FIXME: the follow-on warning should really be suppressed, but apparently
// having an ill-formed return type doesn't mark anything as invalid.
// expected-error@-4 {{can only be applied}}

- (instancetype)o0 __attribute__((__swift_error__(none)));
- (instancetype)o1 __attribute__((__swift_error__(nonnull_error)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a method with an error parameter}}
- (instancetype)o2 __attribute__((__swift_error__(null_result)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a method with an error parameter}}
- (instancetype)o3 __attribute__((__swift_error__(nonzero_result)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a method with an error parameter}}
- (instancetype)o4 __attribute__((__swift_error__(zero_result)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a method with an error parameter}}
@end

extern BOOL m0(CFErrorRef *) __attribute__((__swift_error__(none)));
extern BOOL m1(CFErrorRef *) __attribute__((__swift_error__(nonnull_error)));
extern BOOL m2(CFErrorRef *) __attribute__((__swift_error__(null_result)));
// expected-error@-1 {{'__swift_error__' attribute with 'null_result' convention can only be applied to a function returning a pointer}}
extern BOOL m3(CFErrorRef *) __attribute__((__swift_error__(nonzero_result)));
extern BOOL m4(CFErrorRef *) __attribute__((__swift_error__(zero_result)));

extern Undeclared n0(CFErrorRef *) __attribute__((__swift_error__(none)));
// expected-error@-1 {{unknown type name 'Undeclared'}}
extern Undeclared n1(CFErrorRef *) __attribute__((__swift_error__(nonnull_error)));
// expected-error@-1 {{unknown type name 'Undeclared'}}
extern Undeclared n2(CFErrorRef *) __attribute__((__swift_error__(null_result)));
// expected-error@-1 {{unknown type name 'Undeclared'}}
extern Undeclared n3(CFErrorRef *) __attribute__((__swift_error__(nonzero_result)));
// expected-error@-1 {{unknown type name 'Undeclared'}}
extern Undeclared n4(CFErrorRef *) __attribute__((__swift_error__(zero_result)));
// expected-error@-1 {{unknown type name 'Undeclared'}}

extern void *o0(CFErrorRef *) __attribute__((__swift_error__(none)));
extern void *o1(CFErrorRef *) __attribute__((__swift_error__(nonnull_error)));
extern void *o2(CFErrorRef *) __attribute__((__swift_error__(null_result)));
extern void *o3(CFErrorRef *) __attribute__((__swift_error__(nonzero_result)));
// expected-error@-1 {{'__swift_error__' attribute with 'nonzero_result' convention can only be applied to a function returning an integral type}}
extern void *o4(CFErrorRef *) __attribute__((__swift_error__(zero_result)));
// expected-error@-1 {{'__swift_error__' attribute with 'zero_result' convention can only be applied to a function returning an integral type}}

extern void *p0(void) __attribute__((__swift_error__(none)));
extern void *p1(void) __attribute__((__swift_error__(nonnull_error)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a function with an error parameter}}
extern void *p2(void) __attribute__((__swift_error__(null_result)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a function with an error parameter}}
extern void *p3(void) __attribute__((__swift_error__(nonzero_result)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a function with an error parameter}}
extern void *p4(void) __attribute__((__swift_error__(zero_result)));
// expected-error@-1 {{'__swift_error__' attribute can only be applied to a function with an error parameter}}

extern BOOL b __attribute__((__swift_error__(none)));
// expected-error@-1 {{attribute only applies to functions and Objective-C methods}}
