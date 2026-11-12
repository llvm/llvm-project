// RUN: %clang_cc1 -fsyntax-only -verify %s -DNONE -Wno-gnu
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s -DNONE -Wno-gnu
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -DNONE -Wno-gnu

// RUN: %clang_cc1 -fsyntax-only -verify %s -DALL -Wgnu 
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s -DALL -Wgnu 
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -DALL -Wgnu 

// RUN: %clang_cc1 -fsyntax-only -verify %s -DALL -Wno-gnu \
// RUN:   -Wgnu-anonymous-struct -Wredeclared-class-member \
// RUN:   -Wgnu-flexible-array-union-member -Wgnu-folding-constant \
// RUN:   -Wgnu-empty-struct
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s -DALL -Wno-gnu \
// RUN:   -Wgnu-anonymous-struct -Wredeclared-class-member \
// RUN:   -Wgnu-flexible-array-union-member -Wgnu-folding-constant \
// RUN:   -Wgnu-empty-struct
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -DALL -Wno-gnu \
// RUN:   -Wgnu-anonymous-struct -Wredeclared-class-member \
// RUN:   -Wgnu-flexible-array-union-member -Wgnu-folding-constant \
// RUN:   -Wgnu-empty-struct

// RUN: %clang_cc1 -fsyntax-only -verify %s -DNONE -Wgnu \
// RUN:   -Wno-gnu-anonymous-struct -Wno-redeclared-class-member \
// RUN:   -Wno-gnu-flexible-array-union-member -Wno-gnu-folding-constant \
// RUN:   -Wno-gnu-empty-struct
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s -DNONE -Wgnu \
// RUN:   -Wno-gnu-anonymous-struct -Wno-redeclared-class-member \
// RUN:   -Wno-gnu-flexible-array-union-member -Wno-gnu-folding-constant \
// RUN:   -Wno-gnu-empty-struct
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -DNONE -Wgnu \
// RUN:   -Wno-gnu-anonymous-struct -Wno-redeclared-class-member \
// RUN:   -Wno-gnu-flexible-array-union-member -Wno-gnu-folding-constant \
// RUN:   -Wno-gnu-empty-struct

// Additional disabled tests:
// %clang_cc1 -fsyntax-only -verify %s -DANONYMOUSSTRUCT -Wno-gnu -Wgnu-anonymous-struct
// %clang_cc1 -fsyntax-only -verify %s -DREDECLAREDCLASSMEMBER -Wno-gnu -Wredeclared-class-member
// %clang_cc1 -fsyntax-only -verify %s -DFLEXIBLEARRAYUNIONMEMBER -Wno-gnu -Wgnu-flexible-array-union-member
// %clang_cc1 -fsyntax-only -verify %s -DFOLDINGCONSTANT -Wno-gnu -Wgnu-folding-constant
// %clang_cc1 -fsyntax-only -verify %s -DEMPTYSTRUCT -Wno-gnu -Wgnu-empty-struct

#if NONE
// expected-no-diagnostics
#endif


#if ALL || ANONYMOUSSTRUCT
// expected-warning@+5 {{anonymous structs are a GNU extension}}
#endif

struct as {
  int x;
  struct {
    int a;
    float b;
  };
};


#if ALL || REDECLAREDCLASSMEMBER
// expected-note@+6 {{previous declaration is here}}
// expected-warning@+6 {{class member cannot be redeclared}}
#endif

namespace rcm {
  class A {
    class X;
    class X;
    class X {};
  };
}


#if ALL || FLEXIBLEARRAYUNIONMEMBER
// expected-warning@+6 {{flexible array member 'c1' in a union is a GNU extension}}
#endif

struct faum {
   int l;
   union {
       int c1[];
   };
};


#if (ALL || FOLDINGCONSTANT) && (__cplusplus <= 199711L) // C++03 or earlier modes
// expected-warning@+4 {{in-class initializer for static data member is not a constant expression; folding it to a constant is a GNU extension}}
#endif

struct fic {
  static const int B = int(0.75 * 1000 * 1000);
};


#if ALL || EMPTYSTRUCT
// expected-warning@+3 {{flexible array member 'a' in otherwise empty struct is a GNU extension}}
#endif

struct ofam {int a[];};

