
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -verify %s

#include <ptrcheck.h>

//--------------------------------------------------------------------------
// Functions taking a `__bidi_indexable` pointer argument
//
// Each function should only have one use and the argument name should be
// unique so that we can uniquely identify the warning that these notes
// came from.
//--------------------------------------------------------------------------
// expected-note@+1{{passing argument to parameter 'b_arg0' here}}
void take_bidi0(int *__bidi_indexable b_arg0);
// expected-note@+1{{passing argument to parameter 'b_arg1' here}}
void take_bidi1(int *__bidi_indexable b_arg1);
// expected-note@+1{{passing argument to parameter 'b_arg2' here}}
void take_bidi2(int *__bidi_indexable b_arg2);
// expected-note@+1{{passing argument to parameter 'b_arg3' here}}
void take_bidi3(int *__bidi_indexable b_arg3);
// expected-note@+1{{passing argument to parameter 'b_arg4' here}}
void take_bidi4(int *__bidi_indexable b_arg4);
void take_bidi5(int *__bidi_indexable b_arg5);

//--------------------------------------------------------------------------
// Functions taking an `__indexable` pointer argument
//
// Each function should only have one use and the argument name should be
// unique so that we can uniquely identify the warning that these notes
// came from.
//--------------------------------------------------------------------------
// expected-note@+1{{passing argument to parameter 'i_arg0' here}}
void take_idx0(int *__indexable i_arg0);
// expected-note@+1{{passing argument to parameter 'i_arg1' here}}
void take_idx1(int *__indexable i_arg1);
// expected-note@+1{{passing argument to parameter 'i_arg2' here}}
void take_idx2(int *__indexable i_arg2);
// expected-note@+1{{passing argument to parameter 'i_arg3' here}}
void take_idx3(int *__indexable i_arg3);
// expected-note@+1{{passing argument to parameter 'i_arg4' here}}
void take_idx4(int *__indexable i_arg4);
void take_idx5(int *__indexable i_arg5);

//--------------------------------------------------------------------------
// Pointers marked explicitly as __single
//
// These pointers should only have one usage so that we know that the
// generated notes come from the correct warning.
//--------------------------------------------------------------------------
extern int *__single explicitly_single0;
// expected-note@+1{{pointer 'explicitly_single1' declared here}}
extern int *__single explicitly_single1;
extern int *__single explicitly_single2;
// expected-note@+1{{pointer 'explicitly_single3' declared here}}
extern int *__single explicitly_single3;
extern int *__single explicitly_single4;
// expected-note@+1{{pointer 'explicitly_single5' declared here}}
extern int *__single explicitly_single5;
extern int *__single explicitly_single6;
// expected-note@+1{{pointer 'explicitly_single7' declared here}}
extern int *__single explicitly_single7;
extern int *__single explicitly_single8;
extern int *__single explicitly_single9;
// expected-note@+1{{pointer 'explicitly_single10' declared here}}
extern int *__single explicitly_single10;
extern int *__single explicitly_single11;
// expected-note@+1{{pointer 'explicitly_single12' declared here}}
extern int *__single explicitly_single12;
extern int *__single explicitly_single13;
// expected-note@+1{{pointer 'explicitly_single14' declared here}}
extern int *__single explicitly_single14;
extern int *__single explicitly_single15;
// expected-note@+1{{pointer 'explicitly_single16' declared here}}
extern int *__single explicitly_single16;
extern int *__single explicitly_single17;
// expected-note@+1{{pointer 'explicitly_single18' declared here}}
extern int *__single explicitly_single18;
extern int *__single explicitly_single19;
// expected-note@+1{{pointer 'explicitly_single20' declared here}}
extern int *__single explicitly_single20;
extern int *__single explicitly_single21;
// expected-note@+1{{pointer 'explicitly_single22' declared here}}
extern int *__single explicitly_single22;
extern int *__single explicitly_single23;
// expected-note@+1{{pointer 'explicitly_single24' declared here}}
extern int *__single explicitly_single24;
extern int *__single explicitly_single25;

//--------------------------------------------------------------------------
// Pointers marked implicitly as __single
//
// These pointers should only have one usage so that we know that the
// generated notes come from the correct warning.
//--------------------------------------------------------------------------
extern int *implicitly_single0;
// expected-note@+1{{pointer 'implicitly_single1' declared here}}
extern int *implicitly_single1;
extern int *implicitly_single2;
// expected-note@+1{{pointer 'implicitly_single3' declared here}}
extern int *implicitly_single3;
extern int *implicitly_single4;
// expected-note@+1{{pointer 'implicitly_single5' declared here}}
extern int *implicitly_single5;
extern int *implicitly_single6;
// expected-note@+1{{pointer 'implicitly_single7' declared here}}
extern int *implicitly_single7;
extern int *implicitly_single8;
extern int *implicitly_single9;
// expected-note@+1{{pointer 'implicitly_single10' declared here}}
extern int *implicitly_single10;
extern int *implicitly_single11;
// expected-note@+1{{pointer 'implicitly_single12' declared here}}
extern int *implicitly_single12;
extern int *implicitly_single13;
// expected-note@+1{{pointer 'implicitly_single14' declared here}}
extern int *implicitly_single14;
extern int *implicitly_single15;
// expected-note@+1{{pointer 'implicitly_single16' declared here}}
extern int *implicitly_single16;
extern int *implicitly_single17;
// expected-note@+1{{pointer 'implicitly_single18' declared here}}
extern int *implicitly_single18;
extern int *implicitly_single19;
// expected-note@+1{{pointer 'implicitly_single20' declared here}}
extern int *implicitly_single20;
extern int *implicitly_single21;
// expected-note@+1{{pointer 'implicitly_single22' declared here}}
extern int *implicitly_single22;
extern int *implicitly_single23;
// expected-note@+1{{pointer 'implicitly_single24' declared here}}
extern int *implicitly_single24;
extern int *implicitly_single25;

//--------------------------------------------------------------------------
// Pointers that will cause a bitcast as well as a -fbounds-safety attribute cast.
//
// These pointers should only have one usage so that we know that the
// generated notes come from the correct warning.
//--------------------------------------------------------------------------
typedef unsigned long uint64_t;
_Static_assert(sizeof(uint64_t) > sizeof(int), "type must be larger than int");
extern uint64_t *__single bigger_explicit_single0;
// expected-note@+1{{pointer 'bigger_explicit_single1' declared here}}
extern uint64_t *__single bigger_explicit_single1;
extern uint64_t *__single bigger_explicit_single2;
// expected-note@+1{{pointer 'bigger_explicit_single3' declared here}}
extern uint64_t *__single bigger_explicit_single3;
extern uint64_t *__single bigger_explicit_single4;
// expected-note@+1{{pointer 'bigger_explicit_single5' declared here}}
extern uint64_t *__single bigger_explicit_single5;
extern uint64_t *__single bigger_explicit_single6;
// expected-note@+1{{pointer 'bigger_explicit_single7' declared here}}
extern uint64_t *__single bigger_explicit_single7;
extern uint64_t *__single bigger_explicit_single8;
extern uint64_t *__single bigger_explicit_single9;
// expected-note@+1{{'bigger_explicit_single10' declared here}}
extern uint64_t *__single bigger_explicit_single10;
extern uint64_t *__single bigger_explicit_single11;
// expected-note@+1{{pointer 'bigger_explicit_single12' declared here}}
extern uint64_t *__single bigger_explicit_single12;
extern uint64_t *__single bigger_explicit_single13;
// expected-note@+1{{pointer 'bigger_explicit_single14' declared here}}
extern uint64_t *__single bigger_explicit_single14;
extern uint64_t *__single bigger_explicit_single15;
// expected-note@+1{{pointer 'bigger_explicit_single16' declared here}}
extern uint64_t *__single bigger_explicit_single16;
extern uint64_t *__single bigger_explicit_single17;
// expected-note@+1{{pointer 'bigger_explicit_single18' declared here}}
extern uint64_t *__single bigger_explicit_single18;
extern uint64_t *__single bigger_explicit_single19;
// expected-note@+1{{pointer 'bigger_explicit_single20' declared here}}
extern uint64_t *__single bigger_explicit_single20;
extern uint64_t *__single bigger_explicit_single21;
// expected-note@+1{{pointer 'bigger_explicit_single22' declared here}}
extern uint64_t *__single bigger_explicit_single22;
extern uint64_t *__single bigger_explicit_single23;
// expected-note@+1{{pointer 'bigger_explicit_single24' declared here}}
extern uint64_t *__single bigger_explicit_single24;
extern uint64_t *__single bigger_explicit_single25;

//--------------------------------------------------------------------------
// SINGLE_EXPR
//
// This macro expands to an expression that isn't a DeclRefExpr but still
// but is still __single. Uses of this macro when implicitly cast to an
// indexable pointer should not suggest adding
// the `__counted_by` attribute.
//--------------------------------------------------------------------------
int dummy_array[] = {0, 1, 2, 3};
// SINGLE_EXPR isn't a DeclRefExpr so when this is implicitly cast we shouldn't
// suggest adding `__counted_by` to non existing DeclRefExpr.
#define SINGLE_EXPR ((int *__single)(&dummy_array[2]))

//--------------------------------------------------------------------------
// Function that return a __single pointer
//
// TODO(dliew): We should teach clang to produce a diagnostic for this case
// that suggests adding an attribute to the function return type
// (rdar://91928583).
//--------------------------------------------------------------------------
int *__single returns_explicit_single(void);

typedef struct {
  int *__bidi_indexable field;
} BidiStruct_t;

typedef struct {
  int *__indexable field;
} IdxStruct_t;

void use(void) {
  //--------------------------------------------------------------------------
  // Initialization
  //--------------------------------------------------------------------------
  int *implicit_bidi = explicitly_single0;  // no warning
  int *implicit_bidi2 = implicitly_single0; // no warning
  int *implicit_bidi3 = SINGLE_EXPR;        // no warning
  // Note: This is a different warning than this test is really intended to test
  // but we test it here for completeness.
  // expected-warning@+1{{incompatible pointer types initializing 'int *__bidi_indexable' with an expression of type 'uint64_t *__single' (aka 'unsigned long *__single')}}
  int *implicit_bidi4 = bigger_explicit_single0;
  int *implicit_bidi5 = returns_explicit_single(); // no warning
  int * _Nonnull implicit_bidi6 = returns_explicit_single(); // no warning

  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single1}}
  int *__bidi_indexable explicit_bidi = explicitly_single1;
  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single1}}
  int *__bidi_indexable explicit_bidi2 = implicitly_single1;
  // expected-warning-re@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  int *__bidi_indexable explicit_bidi3 = SINGLE_EXPR;
  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'uint64_t *__single' (aka 'unsigned long *__single') results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single1'}}
  int *__bidi_indexable explicit_bidi4 = bigger_explicit_single1;
  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
  int *__bidi_indexable explicit_bidi5 = returns_explicit_single();
  int *__bidi_indexable explicit_bidi6 = (int *__bidi_indexable)explicitly_single2;         // no warning
  int *__bidi_indexable explicit_bidi7 = (int *__bidi_indexable)implicitly_single2;         // no warning
  int *__bidi_indexable explicit_bidi8 = (int *__bidi_indexable)SINGLE_EXPR;                // no warning
  int *__bidi_indexable explicit_bidi9 = (int *__bidi_indexable)bigger_explicit_single2;    // no warning
  int *__bidi_indexable explicit_bidi10 = (int *__bidi_indexable)returns_explicit_single(); // no warning

  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single3'}}
  int *__indexable explicit_idx = explicitly_single3;
  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single3'}}
  int *__indexable explicit_idx2 = implicitly_single3;
  // expected-warning-re@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  int *__indexable explicit_idx3 = SINGLE_EXPR;
  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'uint64_t *__single' (aka 'unsigned long *__single') results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single3'}}
  int *__indexable explicit_idx4 = bigger_explicit_single3;
  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced}}
  int *__indexable explicit_idx5 = returns_explicit_single();
  int *__indexable explicit_idx6 = (int *__indexable)explicitly_single4;         // no warning
  int *__indexable explicit_idx7 = (int *__indexable)implicitly_single4;         // no warning
  int *__indexable explicit_idx8 = (int *__indexable)SINGLE_EXPR;                // no warning
  int *__indexable explicit_idx9 = (int *__indexable)bigger_explicit_single4;    // no warning
  int *__indexable explicit_idx10 = (int *__indexable)returns_explicit_single(); // no warning

  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single5'}}
  BidiStruct_t bidiStruct = {.field = explicitly_single5};
  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single5'}}
  BidiStruct_t bidiStruct2 = {.field = implicitly_single5};
  // expected-warning-re@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  BidiStruct_t bidiStruct3 = {.field = SINGLE_EXPR};
  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'uint64_t *__single' (aka 'unsigned long *__single') results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single5'}}
  BidiStruct_t bidiStruct4 = {.field = bigger_explicit_single5};
  // expected-warning@+1{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
  BidiStruct_t bidiStruct5 = {.field = returns_explicit_single()};
  BidiStruct_t bidiStruct6 = {.field = (int *__bidi_indexable)explicitly_single6};         // no warning
  BidiStruct_t bidiStruct7 = {.field = (int *__bidi_indexable)implicitly_single6};         // no warning
  BidiStruct_t bidiStruct8 = {.field = (int *__bidi_indexable)SINGLE_EXPR};                // no warning
  BidiStruct_t bidiStruct9 = {.field = (int *__bidi_indexable)bigger_explicit_single6};    // no warning
  BidiStruct_t bidiStruct10 = {.field = (int *__bidi_indexable)returns_explicit_single()}; // no warning

  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single7'}}
  IdxStruct_t idxStruct = {.field = explicitly_single7};
  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single7'}}
  IdxStruct_t idxStruct2 = {.field = implicitly_single7};
  // expected-warning-re@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  IdxStruct_t idxStruct3 = {.field = SINGLE_EXPR};
  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'uint64_t *__single' (aka 'unsigned long *__single') results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single7'}}
  IdxStruct_t idxStruct4 = {.field = bigger_explicit_single7};
  // expected-warning@+1{{initializing type 'int *__indexable' with an expression of type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced}}
  IdxStruct_t idxStruct5 = {.field = returns_explicit_single()};
  IdxStruct_t idxStruct6 = {.field = (int *__indexable)explicitly_single8};         // no warning
  IdxStruct_t idxStruct7 = {.field = (int *__indexable)implicitly_single8};         // no warning
  IdxStruct_t idxStruct8 = {.field = (int *__indexable)SINGLE_EXPR};                // no warning
  IdxStruct_t idxStruct9 = {.field = (int *__indexable)bigger_explicit_single8};    // no warning
  IdxStruct_t idxStruct10 = {.field = (int *__indexable)returns_explicit_single()}; // no warning

  //--------------------------------------------------------------------------
  // Assignment
  //--------------------------------------------------------------------------
  implicit_bidi = implicitly_single9; // no warning
  implicit_bidi = explicitly_single9; // no warning
  implicit_bidi = SINGLE_EXPR;        // no warning
  // Note: This is a different warning than this test is really intended to test
  // but we test it here for completeness.
  // expected-warning@+1{{incompatible pointer types assigning to 'int *__bidi_indexable' from 'uint64_t *__single' (aka 'unsigned long *__single')}}
  implicit_bidi = bigger_explicit_single9;
  implicit_bidi = returns_explicit_single(); // no warning

  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single10'}}
  explicit_bidi = implicitly_single10;
  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single10'}}
  explicit_bidi = explicitly_single10;
  // expected-warning-re@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  explicit_bidi = SINGLE_EXPR;
  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'uint64_t *__single' (aka 'unsigned long *__single') results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single10'}}
  explicit_bidi = bigger_explicit_single10;
  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
  explicit_bidi = returns_explicit_single();

  explicit_bidi = (int *__bidi_indexable)implicitly_single11;       // no warning
  explicit_bidi = (int *__bidi_indexable)explicitly_single11;       // no warning
  explicit_bidi = (int *__bidi_indexable)SINGLE_EXPR;               // no warning
  explicit_bidi = (int *__bidi_indexable)bigger_explicit_single11;  // no warning
  explicit_bidi = (int *__bidi_indexable)returns_explicit_single(); // no warning

  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single24'}}
  explicit_idx = implicitly_single24;
  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single24'}}
  explicit_idx = explicitly_single24;
  // expected-warning-re@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  explicit_idx = SINGLE_EXPR;
  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'uint64_t *__single' (aka 'unsigned long *__single') results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single24'}}
  explicit_idx = bigger_explicit_single24;
  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced}}
  explicit_idx = returns_explicit_single();

  explicit_idx = (int *__indexable)implicitly_single25;       // no warning
  explicit_idx = (int *__indexable)explicitly_single25;       // no warning
  explicit_idx = (int *__indexable)SINGLE_EXPR;               // no warning
  explicit_idx = (int *__indexable)bigger_explicit_single25;  // no warning
  explicit_idx = (int *__indexable)returns_explicit_single(); // no warning

  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single12'}}
  bidiStruct.field = implicitly_single12;
  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single12'}}
  bidiStruct.field = explicitly_single12;
  // expected-warning-re@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  bidiStruct.field = SINGLE_EXPR;
  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'uint64_t *__single' (aka 'unsigned long *__single') results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single12'}}
  bidiStruct.field = bigger_explicit_single12;
  // expected-warning@+1{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
  bidiStruct.field = returns_explicit_single();
  bidiStruct.field = (int *__bidi_indexable)implicitly_single13;       // no warning
  bidiStruct.field = (int *__bidi_indexable)explicitly_single13;       // no warning
  bidiStruct.field = (int *__bidi_indexable)SINGLE_EXPR;               // no warning
  bidiStruct.field = (int *__bidi_indexable)bigger_explicit_single13;  // no warning
  bidiStruct.field = (int *__bidi_indexable)returns_explicit_single(); // no warning

  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single14'}}
  idxStruct.field = implicitly_single14;
  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single14'}}
  idxStruct.field = explicitly_single14;
  // expected-warning-re@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  idxStruct.field = SINGLE_EXPR;
  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'uint64_t *__single' (aka 'unsigned long *__single') results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single14'}}
  idxStruct.field = bigger_explicit_single14;
  // expected-warning@+1{{assigning to type 'int *__indexable' from type 'int *__single' results in an __indexable pointer that will trap if a non-zero offset is dereferenced}}
  idxStruct.field = returns_explicit_single();
  idxStruct.field = (int *__indexable)implicitly_single15;       // no warning
  idxStruct.field = (int *__indexable)explicitly_single15;       // no warning
  idxStruct.field = (int *__indexable)SINGLE_EXPR;               // no warning
  idxStruct.field = (int *__indexable)bigger_explicit_single15;  // no warning
  idxStruct.field = (int *__indexable)returns_explicit_single(); // no warning

  //--------------------------------------------------------------------------
  // Argument pass
  //--------------------------------------------------------------------------
  // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single16'}}
  take_bidi0(implicitly_single16);
  // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single16'}}
  take_bidi1(explicitly_single16);
  // expected-warning-re@+1{{passing type 'int *__single' to parameter of type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  take_bidi2(SINGLE_EXPR);
  // expected-warning@+1{{passing type 'uint64_t *__single' (aka 'unsigned long *__single') to parameter of type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single16'}}
  take_bidi3(bigger_explicit_single16);
  // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
  take_bidi4(returns_explicit_single());

  take_bidi5((int *__bidi_indexable)implicitly_single17);       // no warning
  take_bidi5((int *__bidi_indexable)explicitly_single17);       // no warning
  take_bidi5((int *__bidi_indexable)SINGLE_EXPR);               // no warning
  take_bidi5((int *__bidi_indexable)bigger_explicit_single17);  // no warning
  take_bidi5((int *__bidi_indexable)returns_explicit_single()); // no warning

  // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single18'}}
  take_idx0(implicitly_single18);
  // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single18'}}
  take_idx1(explicitly_single18);
  // expected-warning-re@+1{{passing type 'int *__single' to parameter of type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  take_idx2(SINGLE_EXPR);
  // expected-warning@+1{{passing type 'uint64_t *__single' (aka 'unsigned long *__single') to parameter of type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single18'}}
  take_idx3(bigger_explicit_single18);
  // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced}}
  take_idx4(returns_explicit_single());

  take_idx5((int *__indexable)implicitly_single19);       // no warning
  take_idx5((int *__indexable)explicitly_single19);       // no warning
  take_idx5((int *__indexable)SINGLE_EXPR);               // no warning
  take_idx5((int *__indexable)bigger_explicit_single19);  // no warning
  take_idx5((int *__indexable)returns_explicit_single()); // no warning
}

//--------------------------------------------------------------------------
// Implicit conversion on return
//--------------------------------------------------------------------------
int *__bidi_indexable warn_ret_bidi_expl(void) {
  // expected-warning@+1{{returning type 'int *__single' from a function with result type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single20'}}
  return explicitly_single20;
}

int *__bidi_indexable warn_ret_bidi_impl(void) {
  // expected-warning@+1{{returning type 'int *__single' from a function with result type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single20'}}
  return implicitly_single20;
}

int *__bidi_indexable warn_ret_bidi_single_expr(void) {
  // expected-warning-re@+1{{returning type 'int *__single' from a function with result type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  return SINGLE_EXPR;
}

int *__bidi_indexable warn_ret_bidi_bigger_explicit(void) {
  // expected-warning@+1{{returning type 'uint64_t *__single' (aka 'unsigned long *__single') from a function with result type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single20'}}
  return bigger_explicit_single20;
}

int *__bidi_indexable warn_ret_bidi_tail_call_ret_single(void) {
  // expected-warning@+1{{returning type 'int *__single' from a function with result type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
  return returns_explicit_single();
}

int *__bidi_indexable no_warn_ret_bidi_expl(void) {
  return (int *__bidi_indexable)explicitly_single21; // no warning
}

int *__bidi_indexable no_warn_ret_bidi_impl(void) {
  return (int *__bidi_indexable)implicitly_single21; // no warning
}

int *__bidi_indexable no_warn_ret_bidi_single_expr(void) {
  return (int *__bidi_indexable)SINGLE_EXPR; // no warning
}

int *__bidi_indexable no_warn_ret_bidi_bigger_explicit(void) {
  return (int *__bidi_indexable)bigger_explicit_single21; // no warning
}

int *__bidi_indexable no_warn_ret_bidi_tail_call_ret_single(void) {
  return (int *__bidi_indexable)returns_explicit_single(); // no warning
}

int *__indexable warn_ret_idx_expl(void) {
  // expected-warning@+1{{returning type 'int *__single' from a function with result type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'explicitly_single22'}}
  return explicitly_single22;
}

int *__indexable warn_ret_idx_impl(void) {
  // expected-warning@+1{{returning type 'int *__single' from a function with result type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'implicitly_single22'}}
  return implicitly_single22;
}

int *__indexable warn_ret_idx_single_expr(void) {
  // expected-warning-re@+1{{returning type 'int *__single' from a function with result type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced{{$}}}}
  return SINGLE_EXPR;
}

int *__indexable warn_ret_idx_bigger_explicit(void) {
  // expected-warning@+1{{returning type 'uint64_t *__single' (aka 'unsigned long *__single') from a function with result type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'bigger_explicit_single22'}}
  return bigger_explicit_single22;
}

int *__indexable warn_ret_idx_tail_call_ret_single(void) {
  // expected-warning@+1{{returning type 'int *__single' from a function with result type 'int *__indexable' results in an __indexable pointer that will trap if a non-zero offset is dereferenced}}
  return returns_explicit_single();
}

int *__indexable no_warn_ret_idx_expl(void) {
  return (int *__indexable)explicitly_single23; // no warning
}

int *__indexable no_warn_ret_idx_impl(void) {
  return (int *__indexable)implicitly_single23; // no warning
}

int *__indexable no_warn_ret_idx_single_expr(void) {
  return (int *__indexable)SINGLE_EXPR; // no warning
}

int *__indexable no_warn_ret_idx_bigger_explicit(void) {
  return (int *__indexable)bigger_explicit_single23; // no warning
}

int *__indexable no_warn_ret_idx_tail_call_ret_single(void) {
  return (int *__indexable)returns_explicit_single(); // no warning
}
