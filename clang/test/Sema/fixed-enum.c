// RUN: %clang_cc1 -Weverything -xc++ -std=c++11 -DCXX11 -verify %s
// RUN: %clang_cc1 -Weverything -xc++ -std=c++03 -DCXX03 -verify %s
// RUN: %clang_cc1 -Weverything -xobjective-c -DOBJC -verify %s
// RUN: %clang_cc1 -Weverything -std=c11 -xc -DC11 -verify %s
// RUN: %clang_cc1 -pedantic    -std=c11 -xc -DC11 -verify %s
// RUN: %clang_cc1 -Weverything -std=c11 -xc -fms-extensions -DMS -verify %s
// RUN: %clang_cc1 -Weverything -std=c2x -xc -DC23 -verify %s
// RUN: %clang_cc1 -pedantic    -std=c2x -xc -DC23 -verify -Wpre-c23-compat %s
// RUN: %clang_cc1 -Weverything -std=c23 -xc -DC23 -verify %s
// RUN: %clang_cc1 -pedantic    -std=c23 -xc -DC23 -verify -Wpre-c23-compat %s
// RUN: %clang_cc1 -Weverything -std=c23 -xc -fms-extensions -DC23 -verify %s

enum X : int {e};
#if defined(CXX11)
// expected-warning@-2{{enumeration types with a fixed underlying type are incompatible with C++98}}
#elif defined(CXX03)
// expected-warning@-4{{enumeration types with a fixed underlying type are a C++11 extension}}
#elif defined(OBJC)
// diagnostic
#elif defined(C23)
// expected-warning@-8{{enumeration types with a fixed underlying type are incompatible with C standards before C23}}
#elif defined(C11)
// expected-warning@-10{{enumeration types with a fixed underlying type are a C23 extension}}
#elif defined(MS)
// expected-warning@-12{{enumeration types with a fixed underlying type are a Microsoft extension}}
#endif

// Don't warn about the forward declaration in any language mode.
enum Fwd : int;
enum Fwd : int { e2 };
#if !defined(OBJC) && !defined(C23)
// expected-warning@-3 {{enumeration types with a fixed underlying type}}
// expected-warning@-3 {{enumeration types with a fixed underlying type}}
#elif defined(C23)
// expected-warning@-6 {{enumeration types with a fixed underlying type are incompatible with C standards before C23}}
// expected-warning@-6 {{enumeration types with a fixed underlying type are incompatible with C standards before C23}}
#endif

// Always error on the incompatible redeclaration.
enum BadFwd : int;
#if !defined(OBJC) && !defined(C23)
// expected-warning@-2 {{enumeration types with a fixed underlying type}}
#elif defined(C23)
// expected-warning@-4 {{enumeration types with a fixed underlying type are incompatible with C standards before C23}}
#endif
// expected-note@-6 {{previous declaration is here}}
enum BadFwd : char { e3 };
#if !defined(OBJC) && !defined(C23)
// expected-warning@-2 {{enumeration types with a fixed underlying type}}
#elif defined(C23)
// expected-warning@-4 {{enumeration types with a fixed underlying type are incompatible with C standards before C23}}
#endif
// expected-error@-6 {{enumeration redeclared with different underlying type 'char' (was 'int')}}
