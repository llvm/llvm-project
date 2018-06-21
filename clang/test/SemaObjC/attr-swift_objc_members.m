// RUN: %clang_cc1 -verify -fsyntax-only %s

#if !__has_attribute(swift_objc_members)
#  error Cannot query presence of swift_objc_members attribute.
#endif

__attribute__((swift_objc_members))
__attribute__((objc_root_class))
@interface A
@end

__attribute__((swift_objc_members)) // expected-error{{'swift_objc_members' attribute only applies to Objective-C interfaces}}
@protocol P
@end

__attribute__((swift_objc_members)) // expected-error{{'swift_objc_members' attribute only applies to Objective-C interfaces}}
extern void foo(void);
