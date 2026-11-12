// Test the __VA_ARGS__ comma swallowing extensions of various compiler dialects.

// RUN: %clang_cc1 -E %s | FileCheck -check-prefix=GCC -strict-whitespace %s
// RUN: %clang_cc1 -E -std=c99 %s | FileCheck -check-prefix=C99 -strict-whitespace %s
// RUN: %clang_cc1 -E -std=c11 %s | FileCheck -check-prefix=C99 -strict-whitespace %s
// RUN: %clang_cc1 -E -x c++ %s | FileCheck -check-prefix=GCC -strict-whitespace %s
// RUN: %clang_cc1 -E -std=gnu99 %s | FileCheck -check-prefix=GCC -strict-whitespace %s
// RUN: %clang_cc1 -E -fms-compatibility %s | FileCheck -check-prefix=MS -strict-whitespace %s
// RUN: %clang_cc1 -E -DNAMED %s | FileCheck -check-prefix=GCC -strict-whitespace %s
// RUN: %clang_cc1 -E -std=c99 -DNAMED %s | FileCheck -check-prefix=C99 -strict-whitespace %s


#ifndef NAMED
# define A(...)   [ __VA_ARGS__ ]
# define B(...)   [ , __VA_ARGS__ ]
# define C(...)   [ , ## __VA_ARGS__ ]
# define D(A,...) [ A , ## __VA_ARGS__ ]
# define E(A,...) [ __VA_ARGS__ ## A ]
#else
// These are the GCC named argument versions of the C99-style variadic macros.
// Note that __VA_ARGS__ *may* be used as the name, this is not prohibited!
# define A(__VA_ARGS__...)   [ __VA_ARGS__ ]
# define B(__VA_ARGS__...)   [ , __VA_ARGS__ ]
# define C(__VA_ARGS__...)   [ , ## __VA_ARGS__ ]
# define D(A,__VA_ARGS__...) [ A , ## __VA_ARGS__ ]
# define E(A,__VA_ARGS__...) [ __VA_ARGS__ ## A ]
#endif


1: A()      B()      C()      D()      E()
2: A(a)     B(a)     C(a)     D(a)     E(a)
3: A(,)     B(,)     C(,)     D(,)     E(,)
4: A(a,b,c) B(a,b,c) C(a,b,c) D(a,b,c) E(a,b,c)
5: A(a,b,)  B(a,b,)  C(a,b,)  D(a,b,)

// The GCC ", ## __VA_ARGS__" extension swallows the comma when followed by
// empty __VA_ARGS__.  This extension does not apply in -std=c99 mode, but
// does apply in C++.
//
// GCC: 1: [ ] [ , ] [ ] [ ] [ ]
// GCC: 2: [ a ] [ , a ] [ ,a ] [ a ] [ a ]
// GCC: 3: [ , ] [ , , ] [ ,, ] [ , ] [ ]
// GCC: 4: [ a,b,c ] [ , a,b,c ] [ ,a,b,c ] [ a ,b,c ] [ b,ca ]
// GCC: 5: [ a,b, ] [ , a,b, ] [ ,a,b, ] [ a ,b, ]

// Under C99 standard mode, the GCC ", ## __VA_ARGS__" extension *does not*
// swallow the comma when followed by empty __VA_ARGS__.
//
// C99: 1: [ ] [ , ] [ , ] [ ] [ ]
// C99: 2: [ a ] [ , a ] [ ,a ] [ a ] [ a ]
// C99: 3: [ , ] [ , , ] [ ,, ] [ , ] [ ]
// C99: 4: [ a,b,c ] [ , a,b,c ] [ ,a,b,c ] [ a ,b,c ] [ b,ca ]
// C99: 5: [ a,b, ] [ , a,b, ] [ ,a,b, ] [ a ,b, ]

// Microsoft's extension is on ", __VA_ARGS__" (without explicit ##) where
// the comma is swallowed when followed by empty __VA_ARGS__.
//
// MS: 1: [ ] [ ] [ ] [ ] [ ]
// MS: 2: [ a ] [ , a ] [ ,a ] [ a ] [ a ]
// MS: 3: [ , ] [ , , ] [ ,, ] [ , ] [ ]
// MS: 4: [ a,b,c ] [ , a,b,c ] [ ,a,b,c ] [ a ,b,c ] [ b,ca ]
// MS: 5: [ a,b, ] [ , a,b, ] [ ,a,b, ] [ a ,b, ]

// FIXME: Item 3(d) in MS output should be [ ] not [ , ]
