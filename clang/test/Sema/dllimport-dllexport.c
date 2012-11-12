// RUN: %clang_cc1 -triple i386-mingw32 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-mingw32 -fsyntax-only -verify %s

inline void __attribute__((dllexport)) foo1(){} // expected-warning{{dllexport attribute ignored}}
inline void __attribute__((dllimport)) foo2(){} // expected-warning{{dllimport attribute ignored}}

void __attribute__((dllimport)) foo3(){} // expected-error{{dllimport attribute can be applied only to symbol declaration}}

void __attribute__((dllimport, dllexport)) foo4(); // expected-warning{{dllimport attribute ignored}}

void __attribute__((dllexport)) foo5();
void __attribute__((dllimport)) foo5(); // expected-warning{{dllimport attribute ignored}}

typedef int __attribute__((dllexport)) type6; // expected-warning{{'dllexport' attribute only applies to variables and functions}}

typedef int __attribute__((dllimport)) type7; // expected-warning{{'dllimport' attribute only applies to variables and functions}}

void __attribute__((dllimport)) foo6();
void foo6(){} // expected-warning {{'foo6' redeclared without dllimport attribute: previous dllimport ignored}}

// PR6269
inline void __declspec(dllexport) foo7(){} // expected-warning{{dllexport attribute ignored}}
inline void __declspec(dllimport) foo8(){} // expected-warning{{dllimport attribute ignored}}

void __declspec(dllimport) foo9(){} // expected-error{{dllimport attribute can be applied only to symbol declaration}}

void __declspec(dllimport) __declspec(dllexport) foo10(); // expected-warning{{dllimport attribute ignored}}

void __declspec(dllexport) foo11();
void __declspec(dllimport) foo11(); // expected-warning{{dllimport attribute ignored}}

typedef int __declspec(dllexport) type1; // expected-warning{{'dllexport' attribute only applies to variables and functions}}

typedef int __declspec(dllimport) type2; // expected-warning{{'dllimport' attribute only applies to variables and functions}}

void __declspec(dllimport) foo12();
void foo12(){} // expected-warning {{'foo12' redeclared without dllimport attribute: previous dllimport ignored}}

void __attribute__((dllimport)) foo13(); // expected-warning{{dllimport attribute ignored}}
void __attribute__((dllexport)) foo13();

extern int foo14 __attribute__((dllexport));
extern int foo14 __attribute__((dllimport));  // expected-warning{{dllimport attribute ignored}}
