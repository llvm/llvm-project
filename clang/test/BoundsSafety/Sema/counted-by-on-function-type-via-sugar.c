// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s

// A bounds attribute (__counted_by / __sized_by / __ended_by and the _or_null
// variants) must not be applied to a function type that is reached through a
// *name* -- a typedef or __typeof__ (or, in C++, a using-alias or decltype).
// Otherwise the attribute is silently sunk onto the function's return pointer
// (Apple clang rejects this; swiftlang stable/23.x asserts). The rejection
// lives in ConstructDynamicBoundType::HandleNamedAliasType, the shared handler
// for such "naming" sugar: a directly-written function type -- bare, or wrapped
// in *transparent* declarator sugar (parentheses, a calling-convention
// attribute, or an attribute spelled via a macro) -- dispatches to its own
// visitor instead and stays valid, so a bounds attribute there still
// legitimately applies to the return type. The check uses the canonical type,
// so it also sees through an AttributedType / MacroQualifiedType sitting
// between the name and the function.

#include <ptrcheck.h>

int g;

#define PRESERVE_MOST __attribute__((preserve_most))

//===--- REJECT: bounds attribute on a function type reached through a name ---===

// (1) plain typedef of a function type
typedef int *fn_t(int);
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
fn_t r_typedef __counted_by(g);

// (2) chain of typedefs -- still reached through a name
typedef fn_t fn2_t;
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
fn2_t r_typedef_chain __counted_by(g);

// (3) __typeof__ of a function
int *some_fn(int);
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
__typeof__(some_fn) r_typeof __counted_by(g);

// (4) an AttributedType (a calling convention) sits between the typedef and the
//     function type; the canonical-type check sees through it.
typedef int *fn_attr_t(int) __attribute__((preserve_most));
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
fn_attr_t r_typedef_attr __counted_by(g);

// (5) a MacroQualifiedType (attribute spelled via a macro) wraps that
//     AttributedType; also seen through.
typedef int *fn_macro_t(int) PRESERVE_MOST;
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
fn_macro_t r_typedef_macro __counted_by(g);

// The other bounds-attribute spellings behave the same way.
// expected-error@+1{{'sized_by' only applies to pointers}}
fn_t r_sized __sized_by(g);
// expected-error@+1{{'counted_by_or_null' only applies to pointers}}
fn_t r_counted_or_null __counted_by_or_null(g);
// expected-error@+1{{'sized_by_or_null' only applies to pointers}}
fn_t r_sized_or_null __sized_by_or_null(g);

#ifdef __cplusplus
// (6) C++ using-alias of a function type -- naming sugar, reject
using fn_using_t = int *(int);
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
fn_using_t r_using __counted_by(g);

// (7) C++ decltype of a function type -- naming sugar, reject
// expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
decltype(some_fn) r_decltype __counted_by(g);
#endif

//===--- ACCEPT: directly-written function types (transparent sugar only) ---===

// (a) bare function with a counted return -- baseline
int *__counted_by(len) ok_direct(int len);

// (b) ParenType from declarator parentheses around the typedef name
typedef int *__counted_by(len) (ok_paren_t)(int len);

// (c) AttributedType (calling convention) on a directly-written function
int *__attribute__((preserve_most)) __counted_by(len) ok_attr(int len);

// (d) MacroQualifiedType (attribute macro) on a directly-written function
int *PRESERVE_MOST ok_macro(int len) __counted_by(len);

#ifdef __cplusplus
// (e) C++ decltype of a pointer return type on a directly-written function
decltype(some_fn(0)) ok_decltype(int len) __counted_by(len);
#endif
