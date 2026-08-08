// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -std=c++17 -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -std=c++17 -x objective-c++ -verify %s

// C++-only named-alias sugar over which a bounds attribute is applied: a
// using-alias declaration (`using X = T;`, a TypedefType), a using-declaration
// (`using ns::X;`, a UsingType), and `decltype`. Companion to
// counted-by-named-alias-sugar.c (typedef / __typeof__, exercised in every
// mode). See ConstructDynamicBoundType in SemaDeclAttr.cpp.

#include <ptrcheck.h>

extern int *gp;
extern int gi;

//===--- ACCEPT: bounds attribute reaches a pointer through a named alias ---===

// (1) using-alias declaration -> TypedefType
using ualias_t = int *;
struct GoodUsingAlias {
  int n;
  ualias_t buf __counted_by(n);
};

// (2) using-declaration bringing a typedef into scope -> UsingType
namespace ns { typedef int *ptr_to_int_t; }
using ns::ptr_to_int_t;
struct GoodUsingDecl {
  int n;
  ptr_to_int_t buf __counted_by(n);
};

// (3) decltype -> DecltypeType
struct GoodDecltype {
  int n;
  decltype(gp) buf __counted_by(n);
};

//===--- REJECT: bounds attribute on a non-pointer reached through a name ---===

// (4) decltype of a non-pointer expression
struct BadDecltype {
  int n;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  decltype(gi) buf __counted_by(n);
};

//===--- Dependent-operand probe (documents no-crash behavior) ---------------===

// A dependent `decltype`/`__typeof__` operand is not resolved at template
// definition, so the bounds attribute is deferred (the field type stays
// `decltype(member)` with no CountAttributedType built) and the named-alias
// visitor never sees a dependent, self-desugaring node. Instantiation on a
// concrete pointer type then builds the attribute normally. Neither path
// diagnoses or crashes.
template <class T>
struct Dependent {
  T member;
  int n;
  decltype(member) dbuf __counted_by(n);
};
template struct Dependent<int *>;
