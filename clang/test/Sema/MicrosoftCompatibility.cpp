// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-compatibility

// PR15845
int foo(xxx); // expected-error{{unknown type name}}

struct cls {
  char *m;
};

char * cls::* __uptr wrong2 = &cls::m; // expected-error {{'__uptr' attribute cannot be used with pointers to members}}

// Microsoft allows inline, __inline, and __forceinline to appear on a typedef
// of a function type, but only in C. See GitHub #124869 for more details.
typedef int inline Foo1(int);       // expected-error {{'inline' can only appear on functions}}
typedef int __inline Foo2(int);     // expected-error {{'inline' can only appear on functions}}
typedef int __forceinline Foo(int); // expected-error {{'inline' can only appear on functions}} \
                                       expected-warning {{'__forceinline' attribute only applies to functions and statements}}
