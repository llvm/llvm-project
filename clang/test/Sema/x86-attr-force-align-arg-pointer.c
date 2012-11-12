// RUN: %clang_cc1 -triple i386-apple-darwin10 -fsyntax-only -verify %s

int a __attribute__((force_align_arg_pointer)); // expected-warning{{attribute only applies to functions}}

// It doesn't matter where the attribute is located.
void b(void) __attribute__((force_align_arg_pointer));
void __attribute__((force_align_arg_pointer)) c(void);

// Functions only have to be declared force_align_arg_pointer once.
void b(void) {}

// It doesn't matter which declaration has the attribute.
void d(void);
void __attribute__((force_align_arg_pointer)) d(void) {}

// Attribute is ignored on function pointer types.
void (__attribute__((force_align_arg_pointer)) *p)();
typedef void (__attribute__((__force_align_arg_pointer__)) *p2)();
// Attribute is also ignored on function typedefs.
typedef void __attribute__((force_align_arg_pointer)) e(void);

