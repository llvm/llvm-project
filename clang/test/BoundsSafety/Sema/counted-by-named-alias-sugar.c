// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

// A bounds attribute (__counted_by / __sized_by and the _or_null variants) is
// applied to the type it names, seeing through *named alias* sugar -- a
// typedef, __typeof__, or (in C++) a using-alias / using-declaration. When the
// named type is a pointer the attribute applies to that pointer; when it is not
// a pointer/array the attribute is rejected. This exercises
// ConstructDynamicBoundType's handling of each named-alias sugar kind (the C++
// -only spellings are in the companion .cpp file).

#include <ptrcheck.h>

typedef int *ptr_to_int_t;
typedef int int_t;
extern int *gp;
extern int gi;

//===--- ACCEPT: bounds attribute reaches a pointer through a named alias ---===

// (1) typedef of a pointer
struct GoodTypedef {
  int n;
  ptr_to_int_t buf __counted_by(n);
};

// (2) __typeof__ of a pointer-typed expression
struct GoodTypeofExpr {
  int n;
  __typeof__(gp) buf __counted_by(n);
};

// (3) __typeof__ of a pointer type
struct GoodTypeofType {
  int n;
  __typeof__(int *) buf __counted_by(n);
};

// (4) chain of typedefs -- still reaches the pointer
typedef ptr_to_int_t nested_ptr_to_int_t;
struct GoodNestedTypedef {
  int n;
  nested_ptr_to_int_t buf __counted_by(n);
};

// The other bounds-attribute spellings behave the same way through a typedef.
struct GoodSpellings {
  int n;
  ptr_to_int_t b_sized __sized_by(n);
  ptr_to_int_t b_counted_or_null __counted_by_or_null(n);
  ptr_to_int_t b_sized_or_null __sized_by_or_null(n);
};

//===--- REJECT: bounds attribute on a non-pointer reached through a name ---===

// (5) typedef of a non-pointer
struct BadTypedef {
  int n;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int_t buf __counted_by(n);
};

// (6) __typeof__ of a non-pointer expression
struct BadTypeofExpr {
  int n;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __typeof__(gi) buf __counted_by(n);
};

// (7) __typeof__ of a non-pointer type
struct BadTypeofType {
  int n;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __typeof__(int) buf __counted_by(n);
};
