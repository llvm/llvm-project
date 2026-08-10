// RUN: %clang_cc1 -verify -fsyntax-only %s

#if !__has_attribute(swift_objc_members)
#error cannot verify presence of swift_objc_members attribute
#endif

__attribute__((__swift_objc_members__))
__attribute__((__objc_root_class__))
@interface I
@end

__attribute__((swift_objc_members))
@protocol P
@end
// expected-error@-3 {{'swift_objc_members' attribute only applies to Objective-C interfaces}}

__attribute__((swift_objc_members))
extern void f(void);
// expected-error@-2 {{'swift_objc_members' attribute only applies to Objective-C interfaces}}

// expected-error@+1 {{'__swift_objc_members__' attribute takes no arguments}}
__attribute__((__swift_objc_members__("J")))
@interface J
@end
