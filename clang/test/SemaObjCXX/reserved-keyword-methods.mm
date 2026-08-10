// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

#define FOR_EACH_KEYWORD(macro) \
macro(asm) \
macro(bool) \
macro(catch) \
macro(class) \
macro(const_cast) \
macro(delete) \
macro(dynamic_cast) \
macro(explicit) \
macro(export) \
macro(false) \
macro(friend) \
macro(mutable) \
macro(namespace) \
macro(new) \
macro(operator) \
macro(private) \
macro(protected) \
macro(public) \
macro(reinterpret_cast) \
macro(static_cast) \
macro(template) \
macro(this) \
macro(throw) \
macro(true) \
macro(try) \
macro(typename) \
macro(typeid) \
macro(using) \
macro(virtual) \
macro(wchar_t)


#define DECLARE_METHOD(name) - (void)name;
#define DECLARE_PROPERTY_WITH_GETTER(name) @property (getter=name) int prop_##name;
@interface A 
//FOR_EACH_KEYWORD(DECLARE_METHOD)
FOR_EACH_KEYWORD(DECLARE_PROPERTY_WITH_GETTER)
@end

