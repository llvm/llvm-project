// RUN: %clang_cc1 -verify -fsyntax-only %s

@interface NSString
+(instancetype)stringWithCString:(const char *)cstr __attribute__((format_arg(1)));
+(instancetype)stringWithString:(NSString *)cstr __attribute__((format_arg(1)));
@end

@protocol MaybeString
-(instancetype)maybeString:(const char *)cstr __attribute__((format_arg(1))); // expected-error {{function does not return string type}}
@end

@class NSAttributedString;

extern NSString *fa2 (const NSString *) __attribute__((format_arg(1)));
extern NSString *fa3 (NSString *) __attribute__((format_arg(1)));

extern void fc1 (const NSString *) __attribute__((format_arg));  // expected-error {{'format_arg' attribute takes one argument}}
extern void fc2 (const NSString *) __attribute__((format_arg())); // expected-error {{'format_arg' attribute takes one argument}}
extern void fc3 (const NSString *) __attribute__((format_arg(1, 2))); // expected-error {{'format_arg' attribute takes one argument}}

struct s1 { int i; } __attribute__((format_arg(1)));  // expected-warning {{'format_arg' attribute only applies to Objective-C methods and non-K&R-style functions}}
union u1 { int i; } __attribute__((format_arg(1)));  // expected-warning {{'format_arg' attribute only applies to}}
enum e1 { E1V0 } __attribute__((format_arg(1))); // expected-warning {{'format_arg' attribute only applies to}}

extern NSString *ff3 (const NSString *) __attribute__((format_arg(3-2)));
extern NSString *ff4 (const NSString *) __attribute__((format_arg(foo))); // expected-error {{use of undeclared identifier 'foo'}}

/* format_arg formats must take and return a string.  */
extern NSString *fi0 (int) __attribute__((format_arg(1)));  // expected-error {{format argument not a string type}}
extern NSString *fi1 (NSString *) __attribute__((format_arg(1))); 

extern NSString *fi2 (NSString *) __attribute__((format_arg(1))); 

extern int fi3 (const NSString *) __attribute__((format_arg(1)));  // expected-error {{function does not return NSString}}
extern NSString *fi4 (const NSString *) __attribute__((format_arg(1))); 
extern NSString *fi5 (const NSString *) __attribute__((format_arg(1))); 

extern NSAttributedString *fattrs (const NSString *) __attribute__((format_arg(1)));
