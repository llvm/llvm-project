

// -triple arm64-apple-darwin23.2.0 is used because some diagnostic text mentions platform specific type sizes
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fblocks -Wno-gnu-alignof-expression -triple arm64-apple-darwin23.2.0 -verify %s

#include <ptrcheck.h>

//==============================================================================
// Parameter source of __single
//==============================================================================

void implicit_unconditional_trap(int *p, int count) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  for (int i = 0; i < count; ++i)
    local[i] = 0; // expected-warning{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 1 or < 0}}
}

// Currently we warn here but we may want to use an explicit cast as a suppression mechanism in the future.
void explicit_single_param(int * __single p, int count) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  for (int i = 0; i < count; ++i)
    local[i] = 0; // expected-warning{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 1 or < 0}}
}

void explicit_single_param_cast(int * __single p, int count) { // expected-note{{pointer 'p' declared here}}
  int *local = (int* __bidi_indexable) p; // expected-note{{pointer 'local' initialized here}}
  for (int i = 0; i < count; ++i)
    local[i] = 0; // expected-warning{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 1 or < 0}}
}

// Don't duplicate the existing warning for explicitly __bidi_indexable pointers.
void explicit_unconditional_trap(int *p, int count) { // expected-note{{pointer 'p' declared here}}
  int *__bidi_indexable local = p; // expected-warning{{initializing type 'int *__bidi_indexable' with an expression of type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'p'}}
  for (int i = 0; i < count; ++i)
    local[i] = 0;
}

// Don't duplicate the existing warning for explicitly __bidi_indexable pointers.
void explicit_unconditional_trap_assign(int *p, int count) { // expected-note{{pointer 'p' declared here}}
  int *__bidi_indexable local;
  local = p; // expected-warning{{assigning to type 'int *__bidi_indexable' from type 'int *__single' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'p'}}
  for (int i = 0; i < count; ++i)
    local[i] = 0;
}

// The parameter is properly annotated, no need to warn.
void counted_by_param(int *__counted_by(count) p, int count) {
  int *local = p;
  for (int i = 0; i < count; ++i)
    local[i] = 0; // no-warning
}

// The parameter is properly annotated, no need to warn.
void indexable_param(int *__indexable p, int count) {
  int *local = p;
  for (int i = 0; i < count; ++i)
    local[i] = 0; // no-warning
}

// The parameter is properly annotated, no need to warn.
void bidi_indexable_param(int *__bidi_indexable p, int count) {
  int *local = p;
  for (int i = 0; i < count; ++i)
    local[i] = 0; // no-warning
}

// The parameter is properly annotated, no need to warn.
void null_terminated_param(int *__null_terminated p) {
  int *local = __unsafe_null_terminated_to_indexable(p);
  while (local) {
    *local = 1;
    ++local; // no-warning
  }
}

void implicit_unconditional_trap_Nonnull_attr(int *p, int count) { // expected-note{{pointer 'p' declared here}}
  int * _Nonnull local = p; // expected-note{{pointer 'local' initialized here}}
  for (int i = 0; i < count; ++i)
    local[i] = 0; // expected-warning{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 1 or < 0}}
}

void simple_assignment(int *p, int count) { // expected-note{{pointer 'p' declared here}}
  int *local; // expected-note{{pointer 'local' declared here}}
  local = p; // expected-note{{pointer 'local' assigned here}}
  for (int i = 0; i < count; ++i)
    local[i] = 0; // expected-warning{{indexing into a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in a trap if the index expression is >= 1 or < 0}}
}

void reassignment_after_indexing(int *p, int count) { // expected-note{{pointer 'p' declared here}}
  // expected-note@+1{{__single parameter 'p' assigned to 'local' here}}
  int *local = p; // expected-note{{pointer 'local' declared here}}
  int *__single something_else; // expected-note{{pointer 'something_else' declared here}}
  for (int i = 0; i < count; ++i) {
    // expected-warning@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap if the index expression is >= 1 or < 0}}
    local[i] = 0;
    local = something_else; // expected-note{{__single local variable 'something_else' assigned to 'local' here}}
  }
}

void reassignment_after_indexing_no_warn(int *p, int count) {
  int *local = p; // no-warning
  int *__bidi_indexable something_else;
  for (int i = 0; i < count; ++i) {
    local[i] = 0;
    local = something_else; // This value isn't a __single
  }
}

void unary_op(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  ++local; // expected-warning-re{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
}

void unary_op2(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  --local; // expected-warning-re{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
}

void unary_op3(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  local++; // expected-warning-re{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
}

void unary_op4(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  local--; // expected-warning-re{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
}


void binary_op(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  *(local + 2) = 3; // expected-warning{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
}

void binary_op2(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  *(2 + local) = 3; // expected-warning{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
}

void binary_op3(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  int* new_p = local + 2;
}

void binary_op4(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  *(local - 1) = 3; // expected-warning{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
}


void binary_op_non_constant_offset(int *p, int offset) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer if the offset is >= 1 or < 0}}
  int* new_p = local + offset;
}

void binary_op_constant_offset(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p;
  int* new_p;
  new_p = local + 0; // No warning
  new_p = 0 + local; // No warning

  int *local2 = p; // expected-note{{pointer 'local2' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local2' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  new_p = local2 + 1;
}


void binary_op_compound_constant_offset(int *p) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  local += 2; // expected-warning{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
}

void binary_op_compound_variable_index(int *p, int offset) { // expected-note{{pointer 'p' declared here}}
  int *local = p; // expected-note{{pointer 'local' initialized here}}
  local += offset; // expected-warning{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer if the offset is >= 1 or < 0}}
}

void constant_index(int* p) { // expected-note 2 {{pointer 'p' declared here}}
  int* local = p; // expected-note 2 {{pointer 'local' initialized here}}
  local[0] = 5; // No warning
  local[(int) 0] = 6; // No warning
  local[((int) 0)] = 6; // No warning
  local[1+2+3 - 6] = 6; // No warning
  *local = 10; // No warning

  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 6;

  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1+2+3 -5] = 6;
}

void param_single_paren_used_initialized_arith(int* p) { // expected-note{{pointer 'p' declared here}}
  int* (local) = (p); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  (++local);
}

void param_single_paren_used_initialized_arith2(int* p) { // expected-note{{pointer 'p' declared here}}
  int* (local) = (p); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  (local) += 1;
}

void param_single_paren_used_assign_arith(int* p) { // expected-note{{pointer 'p' declared here}}
  int* (local); // expected-note{{pointer 'local' declared here}}
  (local) = (p); // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in an out-of-bounds pointer}}
  (++local);
}

void param_single_paren_used_assign_idx(int* p) { // expected-note{{pointer 'p' declared here}}
  int* (local); // expected-note{{pointer 'local' declared here}}
  (local) = (p); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

void param_single_paren_used_assign_idx_cast(void* p) { // expected-note{{pointer 'p' declared here}}
  int* (local); // expected-note{{pointer 'local' declared here}}
  (local) = ((int*) p); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

void param_single_paren_used_assign_idx_cast2(void* p) { // expected-note{{pointer 'p' declared here}}
  int* (local); // expected-note{{pointer 'local' declared here}}
  (local) = (int*) (p); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

void param_single_paren_used_assign_idx_cast3(void* p) { // expected-note{{pointer 'p' declared here}}
  int* (local); // expected-note{{pointer 'local' declared here}}
  (local) = ((int*) ((int*) p)); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

struct UnsizedType;
struct SizedType {
  int foo;
};

void warn_param_incomplete_type_to_sized(struct UnsizedType* p) { // expected-note{{pointer 'p' declared here}}
  struct SizedType* local = (struct SizedType*) p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1].foo = 0;
}

void param_single_explicit_bidi_cast(int* p) { // expected-note{{pointer 'p' declared here}}
  int* local = (int* __bidi_indexable) p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

typedef long long int BigTy;
typedef int SmallerTy;

_Static_assert(sizeof(BigTy) > sizeof(SmallerTy), "expected size diff failed");
void param_single_explicit_cast_type_change_second_cast_to_bigger(BigTy* p) {
  // We should not warn here because `local` gets the bounds of a single
  // `BigTy`, not the bounds of a single `SmallerTy`. This is because
  // the BoundsSafetyPointerCast will create the bounds from BigTy and
  // then the BitCast (which does not affect the stored bounds) to SmallerTy
  // happens afterwards. This can be seen from the AST:
  //
  //
  // `-CStyleCastExpr  'SmallerTy *__bidi_indexable' <BitCast>
  //   `-CStyleCastExpr 'BigTy *__bidi_indexable' <BoundsSafetyPointerCast>
  //     `-ImplicitCastExpr  'BigTy *__single' <LValueToRValue> part_of_explicit_cast
  //       `-DeclRefExpr  'BigTy *__single' lvalue ParmVar 'p' 'BigTy *__single'
  //
  //
  // if sizeof(BigTy) == 8 and sizeof(SmallerTy) == 4, then its safe to index
  // local[0] and local[1]
  //
  SmallerTy* local = (SmallerTy* __bidi_indexable)(BigTy* __bidi_indexable) p;
  local[1] = 0; // No warn
}

void param_single_explicit_cast_type_change_second_cast_to_smaller(SmallerTy* p) { // expected-note{{pointer 'p' declared here}}
  // expected-note@+2{{pointer 'local' declared here}}
  // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes}}
  BigTy* local = (BigTy* __bidi_indexable)(SmallerTy* __bidi_indexable) p;
  // expected-warning@+1{{indexing __bidi_indexable 'local' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  local[1] = 0;
}

void param_single_explicit_cast_to_larger_and_attr_change(SmallerTy* p) { // expected-note{{pointer 'p' declared her}}
  // This example traps at runtime because the `local` gets the bounds of a
  // single `BigTy`.
  //
  // `-CStyleCastExpr 0x12e02a528 <col:18, col:44> 'BigTy *__bidi_indexable' <BoundsSafetyPointerCast>
  //   `-ImplicitCastExpr 0x12e02a510 <col:44> 'BigTy *__single' <BitCast> part_of_explicit_cast
  //     `-ImplicitCastExpr 0x12e02a498 <col:44> 'SmallerTy *__single' <LValueToRValue> part_of_explicit_cast
  //       `-DeclRefExpr 0x12e02a460 <col:44> 'SmallerTy *__single' lvalue ParmVar 0x12d90b008 'p' 'SmallerTy *__single'
  //
  BigTy* local = (BigTy* __bidi_indexable) p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbounds-safety-single-to-indexable-bounds-truncated"
void param_single_explicit_cast_to_smaller_and_attr_change(BigTy* p) { // expected-note{{pointer 'p' declared her}}
  // This example traps at runtime because the `(SmallerTy* __bidi_indexable)` explicit
  // cast does the BitCast first, then the BoundsSafetyPointerCast. So `local` gets
  // the bounds of a single `SmallerTy`.
  //
  // `-CStyleCastExpr 'int *__bidi_indexable' <BoundsSafetyPointerCast>
  //   `-ImplicitCastExpr  'int *__single' <BitCast> part_of_explicit_cast
  //     `-ImplicitCastExpr  'long long *__single' <LValueToRValue> part_of_explicit_cast
  //       `-DeclRefExpr  'long long *__single' lvalue ParmVar 0x135115908 'p' 'long long *__single'
  //
  SmallerTy* local = (SmallerTy* __bidi_indexable) p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
void param_single_implicit_cast_to_smaller_and_attr_change(BigTy* p) { // expected-note{{pointer 'p' declared here}}
  // This example traps the BitCast happens before the BoundsSafetyPointerCast so
  // the wide pointer gets the bounds of a single `SmallerTy`.
  //
  // `-ImplicitCastExpr 'SmallerTy *__bidi_indexable' <BoundsSafetyPointerCast>
  //   `-ImplicitCastExpr 'int *__single' <BitCast>
  //     `-ImplicitCastExpr 'BigTy *__single' <LValueToRValue>
  //       `-DeclRefExpr 'BigTy *__single' lvalue ParmVar 'p' 'BigTy *__single'
  //
  SmallerTy* local = p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

void param_single_implicit_cast_to_larger_and_attr_change(SmallerTy* p) { // expected-note{{pointer 'p' declared here}}
  // This example traps the BitCast happens before the BoundsSafetyPointerCast so
  // the wide pointer gets the bounds of a single `BigTy`.
  //
  // `-VarDecl col:10 used local 'BigTy *__bidi_indexable' cinit
  //   `-ImplicitCastExpr  'BigTy *__bidi_indexable' <BoundsSafetyPointerCast>
  //     `-ImplicitCastExpr  'long long *__single' <BitCast>
  //       `-ImplicitCastExpr  'SmallerTy *__single' <LValueToRValue>
  //        `-DeclRefExpr 'SmallerTy *__single' lvalue ParmVar 'p' 'SmallerTy *__single'
  //
  BigTy* local = p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

struct SimpleType {
    int member;
};

struct SameLayoutAsSimpleType {
    int member;
};
_Static_assert(sizeof(struct SimpleType) == sizeof(struct SameLayoutAsSimpleType), "Expected types to have same size");

void param_implicit_cast_to_type_with_same_layout(struct SimpleType* p) { // expected-note{{pointer 'p' declared here}}
  // Check that even though SimpleType and SameLayoutAsSimpleType are not the same
  // that a warning is still emitted.
  //
  struct SameLayoutAsSimpleType* local = p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1].member = 0;
};

void param_explicit_cast_to_type_with_same_layout(struct SimpleType* p) { // expected-note{{pointer 'p' declared here}}
  // Check that even though SimpleType and SameLayoutAsSimpleType are not the same
  // that a warning is still emitted.
  //
  struct SameLayoutAsSimpleType* local = (struct SameLayoutAsSimpleType*) p; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  local[1].member = 0;
};


#pragma clang diagnostic pop
#pragma clang diagnostic pop

void param_multiple_fb_cast(BigTy* p) { // expected-note{{pointer 'p' declared here}}
    // Check we still warn when there are multiple BoundsSafetyPointerCast.
    // 
    // This example traps because only the top BoundsSafetyPointerCast matters.
    // `local` gets the bounds of a single `SmallerTy`.
    //
    // `-CStyleCastExpr  'SmallerTy *__bidi_indexable' <BoundsSafetyPointerCast>
    //   `-CStyleCastExpr 'SmallerTy *__single' <BoundsSafetyPointerCast>
    //     `-ImplicitCastExpr 'SmallerTy *__bidi_indexable' <BitCast> part_of_explicit_cast
    //       `-CStyleCastExpr 'BigTy *__bidi_indexable' <BoundsSafetyPointerCast>
    //         `-ImplicitCastExpr  'BigTy *__single' <LValueToRValue> part_of_explicit_cast
    //           `-DeclRefExpr  'BigTy *__single' lvalue ParmVar 'p' 'BigTy *__single'
    //
    SmallerTy* local = (SmallerTy* __bidi_indexable)(SmallerTy* __single) (BigTy* __bidi_indexable) p; // expected-note{{pointer 'local' initialized here}}
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
    local[1] = 0;
}


// expected-note@+2{{pointer 'p' declared here}}
// expected-note@+1{{pointer 'q' declared here}}
void param_multiple_singles_idx(int* p, int* q) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = p; // expected-note{{__single parameter 'p' assigned to 'local' here}}
  local = q; // expected-note{{__single parameter 'q' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

// expected-note@+2{{pointer 'p' declared here}}
// expected-note@+1{{pointer 'q' declared here}}
void param_multiple_singles_arith(int* p, int* q) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = p; // expected-note{{__single parameter 'p' assigned to 'local' here}}
  local = q; // expected-note{{__single parameter 'q' assigned to 'local' here}}
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
}

void param_multiple_assignments_with_branches(int b,
  // expected-note@+1{{pointer 'p' declared here}}
  int* p,
   // expected-note@+1{{pointer 'q' declared here}}
  int* q) {

  int* local; // expected-note{{pointer 'local' declared here}}
  if (b)
    local = p; // expected-note{{__single parameter 'p' assigned to 'local' here}}
  else
    local = q; // expected-note{{__single parameter 'q' assigned to 'local' here}}

  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

//==============================================================================
// Pointer arithmetic with mismatched pointees
//==============================================================================
typedef char EvenSmallerTy;
_Static_assert(sizeof(EvenSmallerTy) < sizeof(SmallerTy), "unexpected type size diff");
void param_binary_mismatched_pointee_signed_variable_offset(SmallerTy* p, int off) { // expected-note 6{{pointer 'p' declared here}}
  // expected-note@+3 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  // expected-note@+2 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{pointer 'local' declared here}}
  SmallerTy* local = p;

  // Warn even though these could be a false positives because we don't know the
  // value of `offset` at compile time. The warnings try to workaround that by
  // explaining what values of `offset` create an out-of-bounds pointer.
  //
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer if the offset is >= 4 or < 0}}
  *((EvenSmallerTy*)local + off) = 0;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer if the offset is >= 4 or < 0}}
  *((EvenSmallerTy*)local - off) = 0;

  // Warn about the arithmetic because any offset creates an out-of-bounds pointer
  // due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local + off) = 0;
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local - off) = 0;
}

void param_binary_mismatched_pointee_signed_variable_offset_multiple_single_entities(
  SmallerTy* p, // expected-note 6{{pointer 'p' declared here}}
  SmallerTy* q, // expected-note 6{{pointer 'q' declared here}}
  int cond, int off) { 
  
  // expected-note@+3 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  // expected-note@+2 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 6{{pointer 'local' declared here}}
  SmallerTy* local = p;
  if (cond)
    // expected-note@+2 2{{__single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 4{{__single parameter 'q' assigned to 'local' here}}
    local = q;

  // Warn even though these could be a false positives because we don't know the
  // value of `offset` at compile time. The warnings try to workaround that by
  // explaining what values of `offset` create an out-of-bounds pointer.
  //
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer if the offset is >= 4 or < 0}}
  *((EvenSmallerTy*)local + off) = 0;
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer if the offset is >= 4 or < 0}}
  *((EvenSmallerTy*)local - off) = 0;

  // Warn about the arithmetic because any offset creates an out-of-bounds pointer
  // due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning-re@+2{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local + off) = 0;
  // expected-warning-re@+2{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local - off) = 0;
}

void param_binary_mismatched_pointee_unsigned_variable_offset(SmallerTy* p, unsigned int off) { // expected-note 6{{pointer 'p' declared here}}
  // expected-note@+3 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes}}
  // expected-note@+2 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{pointer 'local' declared here}}
  SmallerTy* local = p;

  // Warn even though this could be a false positive because we don't know the
  // value of `offset` at compile time. The warning tries to workaround that by
  // explaining which values of `offset` create an out-of-bounds pointer.
  //
  // expected-warning-re@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer if the offset is >= 4{{$}}}}
  *((EvenSmallerTy*)local + off) = 0;

  // Warn even though this could be a false positive when `offset == 0`.
  // expected-warning-re@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer if the offset is < 0{{$}}}}}
  *((EvenSmallerTy*)local - off) = 0;

  // Warn about the arithmetic because any offset creates an out-of-bounds pointer
  // due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local + off) = 0;
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local - off) = 0;
}

void param_binary_mismatched_pointee_unsigned_variable_offset_multiple_single_entities(
  SmallerTy* p, // expected-note 6{{pointer 'p' declared here}}
  SmallerTy* q, // expected-note 6{{pointer 'q' declared here}}
  int cond, unsigned int off) { 
  // expected-note@+3 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  // expected-note@+2 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 6{{pointer 'local' declared here}}
  SmallerTy* local = p;
  if (cond)
    // expected-note@+2 2{{__single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
    // expected-note@+1 4{{_single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local = q;

  // Warn even though these could be a false positives because we don't know the
  // value of `offset` at compile time. The warnings try to workaround that by
  // explaining which values of `offset` create an out-of-bounds pointer.
  //
  // expected-warning-re@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer if the offset is >= 4{{$}}}}
  *((EvenSmallerTy*)local + off) = 0;
  // expected-warning-re@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer if the offset is < 0{{$}}}}
  *((EvenSmallerTy*)local - off) = 0;

  // Warn about the arithmetic because any offset creates an out-of-bounds pointer
  // due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning-re@+2{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local + off) = 0;
  // expected-warning-re@+2{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local - off) = 0;
}

void param_binary_mismatched_pointee_const_offset(SmallerTy* p) { // expected-note 12{{pointer 'p' declared here}}
  // expected-note@+3 8{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  // expected-note@+2 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{pointer 'local' declared here}}
  SmallerTy* local = p;
  const int signedOne = 1;
  const int signedFour = 4;

  // Positive or zero effective offsets that won't generate an out-of-bounds pointer
  _Static_assert(sizeof(SmallerTy) == 4, "unexpected size");
  *((EvenSmallerTy*)local + 0) = 0;
  *((EvenSmallerTy*)local + 1) = 0;
  *((EvenSmallerTy*)local + 2) = 0;
  *((EvenSmallerTy*)local + 3) = 0;
  *((EvenSmallerTy*)local + signedOne) = 0;

  *((EvenSmallerTy*)local - -0) = 0;
  *((EvenSmallerTy*)local - -1) = 0;
  *((EvenSmallerTy*)local - -2) = 0;
  *((EvenSmallerTy*)local - -3) = 0;

  // Positive effective offsets that will generate an out-of-bounds pointer
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local + 4) = 0;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local - -4) = 0;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local + signedFour) = 0;

  // Negative effective offsets that will generate an out-of-bounds pointer
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local -1) = 0;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local + -1) = 0;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local - signedOne) = 0;

  // Large negative constant
  const long long int MostNegativeValue = 0x8000000000000000UL;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local + MostNegativeValue) = 0;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *((EvenSmallerTy*)local - MostNegativeValue) = 0;

  // Warn about the arithmetic because any offset creates an out-of-bounds pointer
  // due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local + 0) = 0;
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local - 0) = 0;
}

void param_unary_mismatched_pointee(SmallerTy* p) { // expected-note 4{{pointer 'p' declared here}}
  SmallerTy* local = p;

  // These constructs are forbidden. They are commented out because their
  // presence seems to prevent emission of subsequent diagnostics.
  // error: assignment to cast is illegal, lvalue casts are not supported
  // ((char*) local)++;
  // ++((char*) local);
  // ((char*) local)--;
  // --((char*) local);
  // ((char*) local) += 1;
  // ((char*) local) -= 1;

  // The pointee type of the `p` and `local2` do not match
  // sizeof(*p) > sizeof(*local2)
  EvenSmallerTy* local2 = (EvenSmallerTy* __bidi_indexable)(SmallerTy* __bidi_indexable) p;

  // Don't warn. A single increment won't go out-of-bounds.
  ++local2;
  --local2;
  local2++;
  local2--;

  // There's a false negative here. The analysis isn't tracking the value
  // of pointers as execution progresses so there's no way to detect this right
  // now.
  ++local2;
  ++local2;
  ++local2; // `local2` is now out-of-bounds here.

  // The pointee type of the `p` and `local3` do not match
  // sizeof(*p) < sizeof(*local3). So this pointer points to partially
  // out-of-bounds memory.
  //
  // expected-note@+1 4{{pointer 'local3' initialized here}}
  BigTy* local3 = (BigTy* __bidi_indexable)(SmallerTy* __bidi_indexable) p;

  // expected-warning-re@+1{{pointer arithmetic over a __bidi_indexable local variable 'local3' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
  ++local3;
  // expected-warning-re@+1{{pointer arithmetic over a __bidi_indexable local variable 'local3' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
  --local3;
  // expected-warning-re@+1{{pointer arithmetic over a __bidi_indexable local variable 'local3' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
  local3++;
  // expected-warning-re@+1{{pointer arithmetic over a __bidi_indexable local variable 'local3' initialized from __single parameter 'p' results in an out-of-bounds pointer{{$}}}}
  local3--;
}

void param_unary_mismatched_pointee_multiple_single_entities(SmallerTy* p, // expected-note 4{{pointer 'p' declared here}}
  SmallerTy* q, // expected-note 4{{pointer 'q' declared here}}
  int condition) {
  SmallerTy* local = p;

  if (condition)
    local = q;

  // These constructs are forbidden. They are commented out because their
  // presence seems to prevent emission of subsequent diagnostics.
  // error: assignment to cast is illegal, lvalue casts are not supported
  // ((char*) local)++;
  // ++((char*) local);
  // ((char*) local)--;
  // --((char*) local);
  // ((char*) local) += 1;
  // ((char*) local) -= 1;

  // The pointee type of the `p` and `local2` do not match
  // sizeof(*p) > sizeof(*local2)
  EvenSmallerTy* local2 = (EvenSmallerTy* __bidi_indexable)(SmallerTy* __bidi_indexable) p;

  if (condition)
    local2 = (EvenSmallerTy* __bidi_indexable)(SmallerTy* __bidi_indexable) q;

  // Don't warn. A single increment won't go out-of-bounds.
  ++local2;
  --local2;
  local2++;
  local2--;

  // There's a false negative here. The analysis isn't tracking the value
  // of pointers as execution progresses so there's no way to detect this right
  // now.
  ++local2;
  ++local2;
  ++local2; // `local2` is now out-of-bounds here.

  // The pointee type of the `p` and `local3` do not match
  // sizeof(*p) < sizeof(*local3). So this pointer points to partially
  // out-of-bounds memory.
  //
  // expected-note@+2 4{{pointer 'local3' declared here}}
  // expected-note@+1 4{{__single parameter 'p' assigned to 'local3' here}}
  BigTy* local3 = (BigTy* __bidi_indexable)(SmallerTy* __bidi_indexable) p;

  if (condition)
    // expected-note@+1 4{{__single parameter 'q' assigned to 'local3' here}}
    local3 = (BigTy* __bidi_indexable)(SmallerTy* __bidi_indexable) q;

  // expected-warning-re@+1{{pointer arithmetic over __bidi_indexable local variable 'local3' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  ++local3;
  // expected-warning-re@+1{{pointer arithmetic over __bidi_indexable local variable 'local3' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  --local3;
  // expected-warning-re@+1{{pointer arithmetic over __bidi_indexable local variable 'local3' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  local3++;
  // expected-warning-re@+1{{pointer arithmetic over __bidi_indexable local variable 'local3' that is assigned from a __single pointer results in an out-of-bounds pointer{{$}}}}
  local3--;
}


// Original reproducer for rdar://122055103
typedef struct Header {
    int count;
} Header_t;
typedef struct Nested {
    Header_t header;
    int more_data;
} Nested_t;

void my_memset(void*__sized_by(size), int value, unsigned long long size);

void test(Nested_t* p) {
    Nested_t* local = p;
    // There should be no warning here. This pointer arithmetic is in-bounds.
    my_memset((char*) local + sizeof(Header_t), 0, sizeof(Nested_t) - sizeof(Header_t));
}

//==============================================================================
// Indexing with mismatched pointees
//==============================================================================
typedef char EvenSmallerTy;
_Static_assert(sizeof(EvenSmallerTy) < sizeof(SmallerTy), "unexpected type size diff");
void param_idx_mismatched_pointee_signed_variable_offset(SmallerTy* p, int off) {  // expected-note 6{{pointer 'p' declared here}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  SmallerTy* local = p;

  // Warn even though these could be a false positives because we don't know the
  // value of `off` at compile time. The warnings try to workaround that by
  // explaining what values of `off` create an out-of-bounds pointer.
  //
  // expected-warning@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 4 or < 0}}
  ((EvenSmallerTy*)local)[off] = 0;
  // expected-warning@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 4 or < 0}}
  ((EvenSmallerTy*)local)[-off] = 0;

  // expected-note@+3 4{{pointer 'local2' declared here}}
  // expected-note@+2 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallerTy* local2 = p;

  // Warn about the arithmetic because any offset creates an out-of-bounds
  // pointer due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[off] = 0;
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[-off] = 0;
}


void param_idx_mismatched_pointee_signed_variable_offset_multiple_assignees(
  SmallerTy* p, // expected-note 6{{pointer 'p' declared here}}
  SmallerTy* q, // expected-note 6{{pointer 'q' declared here}}
  int cond,
  int off) {
  // expected-note@+2 2{{pointer 'local' declared here}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  SmallerTy* local = p;
  if (cond)
    // expected-note@+1 2{{__single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
    local = q;

  // Warn even though these could be a false positives because we don't know the
  // value of `off` at compile time. The warnings try to workaround that by
  // explaining what values of `off` create an out-of-bounds pointer.
  //
  // expected-warning@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap if the index expression is >= 4 or < 0}}
  ((EvenSmallerTy*)local)[off] = 0;
  // expected-warning@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap if the index expression is >= 4 or < 0}}
  ((EvenSmallerTy*)local)[-off] = 0;

  // expected-note@+3 4{{pointer 'local2' declared here}}
  // expected-note@+2 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallerTy* local2 = p;
  if (cond)
    // expected-note@+2 2{{__single parameter 'q' assigned to 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 2{{__single parameter 'q' assigned to 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local2 = q;

  // Warn about the arithmetic because any offset creates an out-of-bounds
  // pointer due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[off] = 0;
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[-off] = 0;
}

void param_idx_mismatched_pointee_unsigned_variable_offset(SmallerTy* p, unsigned off, unsigned long long off2) {  // expected-note 7{{pointer 'p' declared here}}
  // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  SmallerTy* local = p;

  // Warn even though these could be a false positives because we don't know the
  // value of `off` at compile time. The warnings try to workaround that by
  // explaining what values of `off` create an out-of-bounds pointer.
  //
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 4{{$}}}}
  ((EvenSmallerTy*)local)[off] = 0;
  
  // The warning text is a little surprising at first glance. One might expect
  // it to say `if the index expression is < 0`. However, the result of
  // negating an unsigned type is still an unsigned type so the warning only
  // says `>= 4`. So while the programmer probably intended a negative offset
  // they will likely get a larger offset than intended.
  //
  // For the first case `-off` will get zero extended after being negated which
  // can result in a large offset. E.g. if `off` is `1` `-off` becomes 2**32 -1
  // (`-1` ins 32-bit two's complement).
  //
  // For the second case `-off` will not get extended after being negated. This
  // can result in an even larger offset. E.g. if `off` is `1` `-off` becomes
  // 2**64 -1 (`-1` in 64-bit two's complement)
  
  // Ideally clang should warn about code like this but it currently doesn't
  // (rdar://123416393).
  //
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 4{{$}}}}
  ((EvenSmallerTy*)local)[-off] = 0;
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 4{{$}}}}
  ((EvenSmallerTy*)local)[-off2] = 0;


  // expected-note@+3 4{{pointer 'local2' declared here}}
  // expected-note@+2 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallerTy* local2 = p;

  // Warn about the arithmetic because any offset creates an out-of-bounds
  // pointer due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[off] = 0;
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[-off] = 0;
}

void param_idx_mismatched_pointee_unsigned_variable_offset_multiple_assignees(
  SmallerTy* p, // expected-note 6{{pointer 'p' declared here}}
  SmallerTy* q, // expected-note 6{{pointer 'q' declared here}}
  int cond,
  unsigned off) {
  // expected-note@+2 2{{pointer 'local' declared here}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  SmallerTy* local = p;
  if (cond)
    // expected-note@+1 2{{__single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
    local = q;

  // Warn even though these could be a false positives because we don't know the
  // value of `off` at compile time. The warnings try to workaround that by
  // explaining what values of `off` create an out-of-bounds pointer.
  //
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap if the index expression is >= 4{{$}}}}
  ((EvenSmallerTy*)local)[off] = 0;
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap if the index expression is >= 4{{$}}}}
  ((EvenSmallerTy*)local)[-off] = 0;

  // expected-note@+3 4{{pointer 'local2' declared here}}
  // expected-note@+2 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes}}
  SmallerTy* local2 = p;
  if (cond)
    // expected-note@+2 2{{__single parameter 'q' assigned to 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 2{{__single parameter 'q' assigned to 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local2 = q;

  // Warn about the arithmetic because any offset creates an out-of-bounds
  // pointer due to `sizeof(BigTy) > sizeof(SmallerTy)`
  //
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[off] = 0;
  // expected-warning@+2{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  // expected-warning@+1{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  ((BigTy*)local2)[-off] = 0;
}

void param_idx_mismatched_pointee_const_offset(SmallerTy* p) { // expected-note 10{{pointer 'p' declared here}}
  // expected-note@+1 6{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  SmallerTy* local = p;
  const int signedOne = 1;
  const int signedFour = 4;

  // Positive or zero effective offsets that won't generate an out-of-bounds pointer
  _Static_assert(sizeof(SmallerTy) == 4, "unexpected size");
  ((EvenSmallerTy*)local)[0] = 0;
  ((EvenSmallerTy*)local)[1] = 0;
  ((EvenSmallerTy*)local)[2] = 0;
  ((EvenSmallerTy*)local)[3] = 0;
  ((EvenSmallerTy*)local)[signedOne] = 0;
  ((EvenSmallerTy*)local)[-0] = 0;

  // Positive effective offsets that will generate an out-of-bounds pointer
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  ((EvenSmallerTy*)local)[4] = 0;
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  ((EvenSmallerTy*)local)[signedFour] = 0;

  // Negative effective offsets that will generate an out-of-bounds pointer
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  ((EvenSmallerTy*)local)[-1] = 0;
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  ((EvenSmallerTy*)local)[-signedOne] = 0;

  // Large negative constant
  const long long int MostNegativeValue = 0x8000000000000000UL;
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap{{$}}}}
  ((EvenSmallerTy*)local)[MostNegativeValue] = 0;
  // expected-warning-re@+3{{overflow in expression; result is {{.+}} with type 'long long'}}
  // expected-warning@+2{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in a trap if the index expression is >= 4 or < 0}}
  // Note: When overflow happens Clang fails to evaluate the index expression as a constant
  ((EvenSmallerTy*)local)[-MostNegativeValue] = 0;

  // expected-note@+3 4{{pointer 'local2' declared here}}
  // expected-note@+2 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes}}
  SmallerTy* local2 = p;

  // Warn about the arithmetic because any offset creates an out-of-bounds pointer
  // due to `sizeof(BigTy) > sizeof(SmallerTy)`
  // expected-warning@+2{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  ((BigTy*)local2)[0] = 0;
  // expected-warning@+2{{indexing __bidi_indexable '((BigTy *)local2)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  ((BigTy*)local2)[-0] = 0;
}

//==============================================================================
// Dereferencing with mismatched pointees
//==============================================================================

void param_deref_mismatched_pointee_smaller(SmallerTy* p, int off) { // expected-note 2{{pointer 'p' declared here}}
  SmallerTy* local = p;
  *((EvenSmallerTy* __single) local) = 0;
  *((EvenSmallerTy*) local) = 0;

  // Check that we don't emit any warnings about the dereferencing
  *(((EvenSmallerTy*) local) + 0) = 0;
  *(((EvenSmallerTy*) local) + 1) = 0;
  *(((EvenSmallerTy*) local) + 2) = 0;
  *(((EvenSmallerTy*) local) + 3) = 0;

  // expected-note@+1{{_single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  SmallerTy* local2 = p;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local2' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  *(((EvenSmallerTy*) local2) + 4) = 0;

  // expected-note@+1{{__single parameter 'p' used to initialize 'local3' here results in 'local3' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local3' to 'EvenSmallerTy *__bidi_indexable' (aka 'char *__bidi_indexable') has pointee type 'EvenSmallerTy' (aka 'char') (1 bytes)}}
  SmallerTy* local3 = p;
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local3' initialized from __single parameter 'p' results in an out-of-bounds pointer if the offset is >= 4 or < 0}}
  *(((EvenSmallerTy*) local3) + off) = 0;
}

void param_deref_mismatched_pointee_larger(SmallerTy* p, int off) { // expected-note 7{{pointer 'p' declared here}}
  // expected-note@+2{{pointer 'local' declared here}}
  // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallerTy* local = p;
  // No warning about dereference as its dereferencing a __single.
  // expected-warning@+1{{explicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
  *((BigTy* __single) local) = 0;

  // expected-note@+3 2{{pointer 'local2' declared here}}
  // expected-note@+2{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but '((BigTy *)local2)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1{{__single parameter 'p' used to initialize 'local2' here results in 'local2' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local2' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallerTy* local2 = p;
  // expected-warning@+2{{dereferencing __bidi_indexable '((BigTy *)local2)' will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local2' is assigned a __single pointer that results in '((BigTy *)local2)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local2' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may tra}}
  *((BigTy*) local2) = 0;

  // expected-note@+2{{pointer 'local3' declared here}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local3' here results in 'local3' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local3' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallerTy* local3 = p;
  // No warning about dereference: There's no warning because the current
  // implementation only warns if the operand of the dereference is a DeclRefExpr
  // after walking through all the casts. This behavior is probably fine because
  // there are already multiple warnings about the bad use of `local3`.
  //
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local3' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local3' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *(((BigTy*) local3) + 0) = 0;

  // expected-note@+2{{pointer 'local4' declared here}}
  // expected-note@+1 2{{__single parameter 'p' used to initialize 'local4' here results in 'local4' having the bounds of a single 'SmallerTy' (aka 'int') (4 bytes) but cast of 'local4' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallerTy* local4 = p;
  // No warning about dereference: There's no warning because the current
  // implementation only warns if the operand of the dereference is a DeclRefExpr
  // after walking through all the casts. This behavior is probably fine because
  // there are already multiple warnings about the bad use of `local4`
  //
  // expected-warning@+2{{pointer arithmetic over a __bidi_indexable local variable 'local4' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local4' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *(((BigTy*) local4) + off) = 0;
}

//==============================================================================
// Unevaluated contexts
//==============================================================================

int* __bidi_indexable bidi_source(void);

void unevaluated_context_sizeof(int *p) { // expected-note{{pointer 'p' declared here}}
  int* local = p;
  int y = sizeof(local[1]);

  // This is very subtle but this checks that variable assignments are ignored in an unevaluated context.
  // if the the assignment to `local2` in `sizeof()` is ignored then the analysis only sees one assignment
  // to `local2` which allows the warning at `local2[1]` to fire. If the assignment to `local2` is not ignored
  // then the warning won't fire.
  int* local2 = p; // expected-note{{pointer 'local2' initialized here}}
  int tmp = sizeof(local2 = bidi_source()); // expected-warning{{expression with side effects has no effect in an unevaluated context}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local2' initialized from __single parameter 'p' results in a trap{{$}}}}
  int z = local2[1];
}

void unevaluated_context_alignof(int *p) { // expected-note{{pointer 'p' declared here}}
  int* local = p;
  int y = _Alignof(local[1]);

  // This is very subtle but this checks that variable assignments are ignored in an unevaluated context.
  // if the the assignment to `local2` in `_Alignof()` is ignored then the analysis only sees one assignment
  // to `local2` which allows the warning at `local2[1]` to fire. If the assignment to `local2` is not ignored
  // then the warning won't fire.
  int* local2 = p; // expected-note{{pointer 'local2' initialized here}}
  int tmp = _Alignof(local2 = bidi_source()); // expected-warning{{expression with side effects has no effect in an unevaluated context}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local2' initialized from __single parameter 'p' results in a trap{{$}}}}
  int z = local2[1];
}

void unevaluated_context___alignof(int *p) { // expected-note{{pointer 'p' declared here}}
  int* local = p;
  int y = __alignof(local[1]);

  // This is very subtle but this checks that variable assignments are ignored in an unevaluated context.
  // if the the assignment to `local2` in `__alignof()` is ignored then the analysis only sees one assignment
  // to `local2` which allows the warning at `local2[1]` to fire. If the assignment to `local2` is not ignored
  // then the warning won't fire.
  int* local2 = p; // expected-note{{pointer 'local2' initialized here}}
  int tmp = __alignof(local2 = bidi_source()); // expected-warning{{expression with side effects has no effect in an unevaluated context}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local2' initialized from __single parameter 'p' results in a trap{{$}}}}
  int z = local2[1];
}

void unevaluated_context_typeof(int *p) { // expected-note{{pointer 'p' declared here}}
  int* local = p;
  typeof(local[1]) y = 0;

  // This is very subtle but this checks that variable assignments are ignored in an unevaluated context.
  // if the the assignment to `local2` in `__typeof()` is ignored then the analysis only sees one assignment
  // to `local2` which allows the warning at `local2[1]` to fire. If the assignment to `local2` is not ignored
  // then the warning won't fire.
  int* local2 = p; // expected-note{{pointer 'local2' initialized here}}
  typeof(local2 = bidi_source()) tmp = 0;
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local2' initialized from __single parameter 'p' results in a trap{{$}}}}
  int z = local2[1];
}

void unevaluated_context__Generic(int *p) { // expected-note 3{{pointer 'p' declared here}}
  // No warning should fire here
  int* local = p; 
  int tmp = _Generic(local[1], // Not evaluated
    int:5, // Only this expr should be evaluated
    char: local[1], // Not evaluated
    default: local[1] // Not evaluated
  );

  // A warning should fire here
  int* local2 = p; // expected-note{{pointer 'local2' initialized here}}
  int tmp2 = _Generic(local2[1], // Not evaluated
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local2' initialized from __single parameter 'p' results in a trap{{$}}}}
    int: local2[1], // Only this expr should be evaluated
    char: 5, // Not evaluated
    default: 5); // Not evaluated

  // A warning should fire here
  int* local3 = p; // expected-note{{pointer 'local3' initialized here}}
  int tmp3 = _Generic(local2[1], // Not evaluated
    char: 5, // Not evaluated
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local3' initialized from __single parameter 'p' results in a trap{{$}}}}
    default: local3[1]); // Only this expr should be evaluated


  // This is very subtle but this checks that variable assignments are ignored in an unevaluated context.
  // if the the assignment to `local4` in `__Generic` is ignored then the analysis only sees one assignment
  // to `local4` which allows the warning at `loca42[1]` to fire. If the assignment to `local4` is not ignored
  // then the warning won't fire.
  int* local4 = p; // expected-note{{pointer 'local4' initialized here}}
  int tmp4;
  _Generic(5,
    int: tmp4 = 0,
    default: local4 = bidi_source()
  );
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local4' initialized from __single parameter 'p' results in a trap{{$}}}}
  int z = local4[1];
}

//==============================================================================
// Blocks
//==============================================================================

// In some senses this is a false positive because the block is never called.
void decl_block(void) {
  int (^blk)(int*) = ^(int* x) { // expected-note{{pointer 'x' declared here}}
    int* block_local = x; // expected-note{{pointer 'block_local' initialized here}}
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'block_local' initialized from __single parameter 'x' results in a trap{{$}}}}
    return block_local[1];
  };
}

void decl_block_with_call(void) {
  int* __single p = 0;
  int (^blk)(int*) = ^(int* x) { // expected-note{{pointer 'x' declared here}}
    int* block_local = x; // expected-note{{pointer 'block_local' initialized here}}
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'block_local' initialized from __single parameter 'x' results in a trap{{$}}}}
    return block_local[1];
  };
  blk(p);
}

void block_capture(int* p) { // expected-note{{pointer 'p' declared here}}
    int (^blk)(void) = ^(void) { 
    int* block_local = p; // expected-note{{pointer 'block_local' initialized here}}
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'block_local' initialized from __single parameter 'p' results in a trap{{$}}}}
    return block_local[1];
  };
}

void block_capture2(int* p) { // expected-note{{pointer 'p' declared here}}
  int* __block f_local; // expected-note{{pointer 'f_local' declared here}}
  int (^blk)(void) = ^(void) { 
    f_local = p; // expected-note{{'f_local' assigned here}}
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'f_local' assigned from __single parameter 'p' results in a trap{{$}}}}
    return f_local[1];
  };
}

void block_capture3(void) {
  int* __single p; // expected-note{{pointer 'p' declared here}}
    int (^blk)(void) = ^(void) { 
    int* block_local = p; // expected-note{{pointer 'block_local' initialized here}}
    // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'block_local' initialized from __single local variable 'p' results in a trap{{$}}}}
    return block_local[1];
  };
}

void block_local(int* p) { // expected-note{{pointer 'p' declared here}}
  int* __block local; // expected-note{{pointer 'local' declared here}}
  local = p; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in a trap{{$}}}}
  local[1] = 0;
}

//==============================================================================
// Global source of __single
//==============================================================================

int* global_count0; // expected-note{{pointer 'global_count0' declared here}}
int* global_count1; // expected-note{{pointer 'global_count1' declared here}}
int* global_count2; // expected-note{{pointer 'global_count2' declared here}}
int* global_count3; // expected-note{{pointer 'global_count3' declared here}}
int* global_count4; // expected-note{{pointer 'global_count4' declared here}}
void* global_count5; // expected-note{{pointer 'global_count5' declared here}}
int* global_count6; // expected-note{{pointer 'global_count6' declared here}}
int* global_count7; // expected-note{{pointer 'global_count7' declared here}}
int* global_count8; // expected-note{{pointer 'global_count8' declared here}}
int* global_count9; // expected-note{{pointer 'global_count9' declared here}}

void initialized_from_single_global_var_idx(void) {
  int* local = global_count0; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single global 'global_count0' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_single_global_var_idx(void) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = global_count1; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single global 'global_count1' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_single_global_var_arith(void) {
  int* local = global_count2; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single global 'global_count2' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_single_global_var_arith(void) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = global_count3; // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single global 'global_count3' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_single_global_var_idx_parens(void) {
  int* local; // expected-note{{pointer 'local' declared here}}
  (local) = (global_count4); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single global 'global_count4' results in a trap{{$}}}}
  (local[1]) = 0;
}

void assigned_from_single_global_var_idx_parens_cast(void) {
  int* local; // expected-note{{pointer 'local' declared here}}
  (local) = ((int*) global_count5); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single global 'global_count5' results in a trap{{$}}}}
  (local[1]) = 0;
}

void assigned_from_multiple_globals_idx(void) {
  // expected-note@+1{{pointer 'local' declared here}}
  int *local = global_count6; // expected-note{{__single global 'global_count6' assigned to 'local' here}}
  local = global_count7; // expected-note{{__single global 'global_count7' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_multiple_globals_arith(void) {
  // expected-note@+1{{pointer 'local' declared here}}
  int *local = global_count8; // expected-note{{__single global 'global_count8' assigned to 'local' here}}
  local = global_count9; // expected-note{{__single global 'global_count9' assigned to 'local' here}}
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
}

//==============================================================================
// Local var source of __single
//==============================================================================

void initialized_from_local_single_var_idx(int* p) {
  int* __single s = p; // expected-note{{pointer 's' declared here}}
  int* local = s; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single local variable 's' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_local_single_var_idx(int* p) {
  int* __single s = p; // expected-note{{pointer 's' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single local variable 's' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_local_single_var_arith(int* p) {
  int* __single s = p; // expected-note{{pointer 's' declared here}}
  int* local = s; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single local variable 's' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_local_single_var_arith(int* p) {
  int* __single s = p; // expected-note{{pointer 's' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s; // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single local variable 's' results in an out-of-bounds pointer}}
  ++local;
}

void initialized_from_local_static_single_var_idx(int* p) {
  static int* __single s = 0; // expected-note{{pointer 's' declared here}}
  int* local = s; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single local variable 's' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_local_static_single_var_idx(int* p) {
  static int* __single s = 0; // expected-note{{pointer 's' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s;  // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single local variable 's' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_local_static_single_var_arith(int* p) {
  static int* __single s = 0; // expected-note{{pointer 's' declared here}}
  int* local = s; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single local variable 's' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_local_static_single_var_arith(int* p) {
  static int* __single s = 0; // expected-note{{pointer 's' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s;  // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single local variable 's' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_local_single_var_idx_parens(int* p) {
  int* __single s = (p); // expected-note{{pointer 's' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}
  (local) = (s); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single local variable 's' results in a trap{{$}}}}
  (local[1]) = 0;
}

void assigned_from_local_single_var_idx_parens_cast(int* p) {
  void* __single s = (p); // expected-note{{pointer 's' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}
  (local) = ((int*) s); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single local variable 's' results in a trap{{$}}}}
  (local[1]) = 0;
}

void initialized_from_multiple_local_single_vars_idx(int* p) {
  int* __single s = p; // expected-note{{pointer 's' declared here}}
  int* __single s2 = p; // expected-note{{pointer 's2' declared here}}
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = s; // expected-note{{__single local variable 's' assigned to 'local' here}}
  local = s2; // expected-note{{__single local variable 's2' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_multiple_local_single_vars_arith(int* p) {
  int* __single s = p; // expected-note{{pointer 's' declared here}}
  int* __single s2 = p; // expected-note{{pointer 's2' declared here}}
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = s; // expected-note{{__single local variable 's' assigned to 'local' here}}
  local = s2; // expected-note{{__single local variable 's2' assigned to 'local' here}}
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
}


//==============================================================================
// function return source of __single
//==============================================================================

int* single_source0(void); // expected-note{{'single_source0' declared here}}
int* single_source1(void); // expected-note{{'single_source1' declared here}}
int* single_source2(void); // expected-note{{'single_source2' declared here}}
int* single_source3(void); // expected-note{{'single_source3' declared here}}
int* single_source4(void); // expected-note{{'single_source4' declared here}}
void* single_source5(void); // expected-note{{'single_source5' declared here}}
int* single_source6(void); // expected-note{{'single_source6' declared here}}
int* single_source7(void); // expected-note{{'single_source7' declared here}}
int* single_source8(void); // expected-note{{'single_source8' declared here}}
int* single_source9(void); // expected-note{{'single_source9' declared here}}

void initialized_from_func_call_return_single_idx(void) {
  int* local = single_source0(); // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single return value from call to 'single_source0' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_func_call_return_single_arith(void) {
  int* local = single_source1(); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single return value from call to 'single_source1' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_func_call_return_single_idx(void) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = single_source2(); // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single return value from call to 'single_source2' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_func_call_return_single_arith(void) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = single_source3(); // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single return value from call to 'single_source3' results in an out-of-bounds pointer}}
  ++local;
}

void initialized_from_func_call_return_single_arith_parens(void) {
  int* local = (single_source4()); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single return value from call to 'single_source4' results in an out-of-bounds pointer}}
  (++local);
}

void initialized_from_func_call_return_single_arith_parens_cast(void) {
  int* local = ((int*) single_source5()); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single return value from call to 'single_source5' results in an out-of-bounds pointer}}
  (++local);
}

// Currently the FunctionDecl for this function actually returns __single (but
// the return value is promoted to __bidi_indexable with the appropriate bounds
// at call sites) so special logic is used to avoid warning for functions like
// these because in practice calls to these functions are not treated as
// returning __single.
void* custom_malloc(unsigned long long size) __attribute__((alloc_size(1)));

void initialized_from_func_call_to_alloc_func_idx(void) {
  int* local = custom_malloc(5);
  local[1] = 0; // No warning
}

void initialized_from_func_call_to_alloc_func_arith(void) {
  int* local = custom_malloc(5);
  ++local; // No warning
}

void assigned_from_func_call_to_alloc_func_idx(void) {
  int* local;
  local = custom_malloc(5);
  local[1] = 0; // No warning
}

void assigned_from_func_call_to_alloc_func_arith(void) {
  int* local;
  local = custom_malloc(5);
  ++local; // No warning
}

void initialized_from_multiple_func_calls_return_single_idx(void) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = single_source6(); // expected-note{{__single return value from call to 'single_source6' assigned to 'local' here}}
  local = single_source7(); // expected-note{{__single return value from call to 'single_source7' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_multiple_func_calls_return_single_artih(void) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = single_source8(); // expected-note{{__single return value from call to 'single_source8' assigned to 'local' here}}
  local = single_source9(); // expected-note{{__single return value from call to 'single_source9' assigned to 'local' here}}
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
}


//==============================================================================
// Array element source of __single
//==============================================================================

void initialized_from_array_elt_idx(int* ptrs[__counted_by(size)], int size) { // expected-note{{'ptrs' declared here}}
  if (size < 2)
    return;
  int* local = ptrs[0]; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single element from array 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_array_elt_arith(int* ptrs[__counted_by(size)], int size) { // expected-note{{'ptrs' declared here}}
  if (size < 2)
    return;
  int* local = ptrs[0]; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single element from array 'ptrs' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_array_elt_idx(int* ptrs[__counted_by(size)], int size) { // expected-note{{'ptrs' declared here}}
  if (size < 2)
    return;
  int* local; // expected-note{{pointer 'local' declared here}}
  local = ptrs[0]; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single element from array 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_array_elt_arith(int* ptrs[__counted_by(size)], int size) { // expected-note{{'ptrs' declared here}}
  if (size < 2)
    return;
  int* local; // expected-note{{pointer 'local' declared here}}
  local = ptrs[0]; // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single element from array 'ptrs' results in an out-of-bounds pointer}}
  ++local;
}

int* global_array0[4]; // expected-note{{'global_array0' declared here}}

void initialized_from_global_array_elt_idx(void) { 
  int* local = global_array0[0]; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single element from array 'global_array0' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_array_elt_arith_parens(int* ptrs[__counted_by(size)], int size) { // expected-note{{'ptrs' declared here}}
  if (size < 2)
    return;
  int* local = (ptrs[0]); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single element from array 'ptrs' results in an out-of-bounds pointer}}
  (++local);
}

void initialized_from_array_elt_arith_parens_cast(int* ptrs[__counted_by(size)], int size) { // expected-note{{'ptrs' declared here}}
  if (size < 2)
    return;
  int* local = ((int*) ptrs[0]); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single element from array 'ptrs' results in an out-of-bounds pointer}}
  (++local);
}

// expected-note@+2{{'ptrs' declared here}}
// expected-note@+1{{'ptrs2' declared here}}
void initialized_from_multiple_array_elt_idx(int* ptrs[__counted_by(size)], int* ptrs2[__counted_by(size)], int size) {
  if (size < 2)
    return;
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = ptrs[0]; // expected-note{{__single element from array 'ptrs' assigned to 'local' here}}
  local = ptrs2[1]; // expected-note{{__single element from array 'ptrs2' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

// expected-note@+2{{'ptrs' declared here}}
// expected-note@+1{{'ptrs2' declared here}}
void initialized_from_multiple_array_elt_arith(int* ptrs[__counted_by(size)], int* ptrs2[__counted_by(size)], int size) {
  if (size < 2)
    return;
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = ptrs[0]; // expected-note{{__single element from array 'ptrs' assigned to 'local' here}}
  local = ptrs2[1]; // expected-note{{__single element from array 'ptrs2' assigned to 'local' here}}
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
}

//==============================================================================
// Struct member source of __single
//==============================================================================

struct StructWithSinglePtr0 {
  int* ptr; // expected-note{{StructWithSinglePtr0::ptr declared here}}
};

struct StructWithSinglePtr1 {
  int* ptr; // expected-note{{StructWithSinglePtr1::ptr declared here}}
};

struct StructWithSinglePtr2 {
  int* ptr; // expected-note{{StructWithSinglePtr2::ptr declared here}}
};


struct StructWithSinglePtr3 {
  int* ptr; // expected-note{{StructWithSinglePtr3::ptr declared here}}
};

struct StructWithSinglePtr4 {
  int* ptr; // expected-note{{StructWithSinglePtr4::ptr declared here}}
};

struct StructWithSinglePtr5 {
  void* ptr; // expected-note{{StructWithSinglePtr5::ptr declared here}}
};

struct StructWithSinglePtr6 {
  int* ptr; // expected-note{{StructWithSinglePtr6::ptr declared here}}
};

struct StructWithSinglePtr7 {
  int* ptr; // expected-note{{StructWithSinglePtr7::ptr declared here}}
};

struct StructWithSinglePtr8 {
  int* ptr; // expected-note{{StructWithSinglePtr8::ptr declared here}}
};

struct StructWithSinglePtr9 {
  int* ptr; // expected-note{{StructWithSinglePtr9::ptr declared here}}
};

union UnionWithSinglePtr0 {
  int* ptr; // expected-note{{UnionWithSinglePtr0::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr1 {
  int* ptr; // expected-note{{UnionWithSinglePtr1::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr2 {
  int* ptr; // expected-note{{UnionWithSinglePtr2::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr3 {
  int* ptr; // expected-note{{UnionWithSinglePtr3::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr4 {
  int* ptr; // expected-note{{UnionWithSinglePtr4::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr5 {
  void* ptr; // expected-note{{UnionWithSinglePtr5::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr6 {
  int* ptr; // expected-note{{UnionWithSinglePtr6::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr7 {
  int* ptr; // expected-note{{UnionWithSinglePtr7::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr8 {
  int* ptr; // expected-note{{UnionWithSinglePtr8::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSinglePtr9 {
  int* ptr; // expected-note{{UnionWithSinglePtr9::ptr declared here}}
  unsigned long long integer;
};

void initialized_from_struct_member_idx(struct StructWithSinglePtr0* s) {
  int* local = s->ptr; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single struct member 'ptr' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_union_member_idx(union UnionWithSinglePtr0* s) {
  int* local = s->ptr; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single union member 'ptr' results in a trap{{$}}}}
  local[1] = 0;
}


void initialized_from_struct_member_arith(struct StructWithSinglePtr1* s) {
  int* local = s->ptr; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single struct member 'ptr' results in an out-of-bounds pointer}}
  ++local;
}

void initialized_from_union_member_arith(union UnionWithSinglePtr1* s) {
  int* local = s->ptr; // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single union member 'ptr' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_struct_member_idx(struct StructWithSinglePtr2* s) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s->ptr; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single struct member 'ptr' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_union_member_idx(union UnionWithSinglePtr2* s) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s->ptr; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single union member 'ptr' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_struct_member_arith(struct StructWithSinglePtr3* s) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s->ptr; // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single struct member 'ptr' results in an out-of-bounds pointer}}
  ++local;
}

void assigned_from_union_member_arith(union UnionWithSinglePtr3* s) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s->ptr; // expected-note{{pointer 'local' assigned here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single union member 'ptr' results in an out-of-bounds pointer}}
  ++local;
}

void initialized_from_struct_member_arith_parens(struct StructWithSinglePtr4* s) {
  int* local = (s->ptr); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single struct member 'ptr' results in an out-of-bounds pointer}}
  (++local);
}

void initialized_from_union_member_arith_parens(union UnionWithSinglePtr4* s) {
  int* local = (s->ptr); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single union member 'ptr' results in an out-of-bounds pointer}}
  (++local);
}

void initialized_from_struct_member_arith_parens_cast(struct StructWithSinglePtr5* s) {
  int* local = ((int*) s->ptr); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single struct member 'ptr' results in an out-of-bounds pointer}}
  (++local);
}

void initialized_from_union_member_arith_parens_cast(union UnionWithSinglePtr5* s) {
  int* local = ((int*) s->ptr); // expected-note{{pointer 'local' initialized here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single union member 'ptr' results in an out-of-bounds pointer}}
  (++local);
}

void initialized_from_multiple_struct_member_idx(struct StructWithSinglePtr6* s1, struct StructWithSinglePtr7* s2) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = s1->ptr; // expected-note{{__single struct member 'ptr' assigned to 'local' here}}
  local = s2->ptr; // expected-note{{__single struct member 'ptr' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_multiple_union_member_idx(union UnionWithSinglePtr6* u1, union UnionWithSinglePtr7* u2) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = u1->ptr; // expected-note{{__single union member 'ptr' assigned to 'local' here}}
  local = u2->ptr; // expected-note{{__single union member 'ptr' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_multiple_struct_member_arith(struct StructWithSinglePtr8* s1, struct StructWithSinglePtr9* s2) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = s1->ptr; // expected-note{{__single struct member 'ptr' assigned to 'local' here}}
  local = s2->ptr; // expected-note{{__single struct member 'ptr' assigned to 'local' here}}
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
}

void initialized_from_multiple_union_member_arith(union UnionWithSinglePtr8* u1, union UnionWithSinglePtr9* u2) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = u1->ptr; // expected-note{{__single union member 'ptr' assigned to 'local' here}}
  local = u2->ptr; // expected-note{{__single union member 'ptr' assigned to 'local' here}}
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
}

struct StructWithArrayOfSingles0 {
  int* ptrs[4]; // expected-note{{StructWithArrayOfSingles0::ptrs declared here}}
};

struct StructWithArrayOfSingles1 {
  int* ptrs[4]; // expected-note{{StructWithArrayOfSingles1::ptrs declared here}}
};

struct StructWithArrayOfSingles2 {
  int* ptrs[4]; // expected-note{{StructWithArrayOfSingles2::ptrs declared here}}
};

struct StructWithArrayOfSingles3 {
  int* ptrs[4]; // expected-note{{StructWithArrayOfSingles3::ptrs declared here}}
};

struct StructWithArrayOfSingles4 {
  int* ptrs[4]; // expected-note{{StructWithArrayOfSingles4::ptrs declared here}}
};

union UnionWithArrayOfSingles0 {
  int* ptrs[4]; // expected-note{{UnionWithArrayOfSingles0::ptrs declared here}}
  int unused;
};

union UnionWithArrayOfSingles1 {
  int* ptrs[4]; // expected-note{{UnionWithArrayOfSingles1::ptrs declared here}}
  int unused;
};

union UnionWithArrayOfSingles2 {
  int* ptrs[4]; // expected-note{{UnionWithArrayOfSingles2::ptrs declared here}}
  int unused;
};

union UnionWithArrayOfSingles3 {
  int* ptrs[4]; // expected-note{{UnionWithArrayOfSingles3::ptrs declared here}}
  int unused;
};

union UnionWithArrayOfSingles4 {
  int* ptrs[4]; // expected-note{{UnionWithArrayOfSingles4::ptrs declared here}}
  int unused;
};

void initialized_from_struct_member_via_array_idx(struct StructWithArrayOfSingles0* s) {
  int* local = s->ptrs[1]; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single struct member 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_struct_member_via_array_idx(struct StructWithArrayOfSingles1* s) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s->ptrs[1]; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single struct member 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

void initialized_from_union_member_via_array_idx(union UnionWithArrayOfSingles0* s) {
  int* local = s->ptrs[1]; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single union member 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_struct_union_via_array_idx(union UnionWithArrayOfSingles1* s) {
  int* local; // expected-note{{pointer 'local' declared here}}
  local = s->ptrs[1]; // expected-note{{pointer 'local' assigned here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' assigned from __single union member 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

struct StructWithArrayOfSingleStructPtrs0 {
  struct StructWithArrayOfSingles2* struct_ptrs[4];
};

void initialized_from_struct_member_via_array_nested_idx(struct StructWithArrayOfSingleStructPtrs0* s) {
  int* local = s->struct_ptrs[1]->ptrs[2]; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single struct member 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

union UnionWithArrayOfSingleUnionPtrs0 {
  union UnionWithArrayOfSingles2* union_ptrs[4];
};

void initialized_from_union_member_via_array_nested_idx(union UnionWithArrayOfSingleUnionPtrs0* s) {
  int* local = s->union_ptrs[1]->ptrs[2]; // expected-note{{pointer 'local' initialized here}}
  // expected-warning-re@+1{{indexing into a __bidi_indexable local variable 'local' initialized from __single union member 'ptrs' results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_multiple_struct_members_via_array_idx(struct StructWithArrayOfSingles3* s1, struct StructWithArrayOfSingles4* s2) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = s1->ptrs[1]; // expected-note{{__single struct member 'ptrs' assigned to 'local' here}}
  local = s2->ptrs[2]; // expected-note{{__single struct member 'ptrs' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

void assigned_from_multiple_union_members_via_array_idx(union UnionWithArrayOfSingles3* u1, union UnionWithArrayOfSingles4* u2) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = u1->ptrs[1]; // expected-note{{__single union member 'ptrs' assigned to 'local' here}}
  local = u2->ptrs[2]; // expected-note{{__single union member 'ptrs' assigned to 'local' here}}
  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

//==============================================================================
// Multiple __single entity types
//==============================================================================
int* global_single_int; // expected-note{{pointer 'global_single_int' declared here}}
int* single_int_source(void); // expected-note{{'single_int_source' declared here}}

struct StructWithSingleIntPtr {
  int* ptr; // expected-note{{StructWithSingleIntPtr::ptr declared here}}
};

struct StructWithSingleIntArray {
  int* ptrs[4]; // expected-note{{StructWithSingleIntArray::ptrs }}
};


union UnionWithSingleIntPtr {
  int* ptr; // expected-note{{UnionWithSingleIntPtr::ptr declared here}}
  unsigned long long integer;
};

union UnionWithSingleIntArray {
  int* ptrs[4]; // expected-note{{UnionWithSingleIntArray::ptrs declared here}}
  unsigned long long integer;
};

void multiple_entity_types(
  // expected-note@+1{{pointer 'p' declared here}}
  int* p,
  // expected-note@+1{{'ptrs' declared here}}
  int* ptrs[__counted_by(size)], int size,
  struct StructWithSingleIntPtr* s,
  union UnionWithSingleIntPtr* u,
  struct StructWithSingleIntArray* sa,
  union UnionWithSingleIntArray* ua) {
  int* local; // expected-note{{pointer 'local' declared here}}
  int* __single local_single; // expected-note{{pointer 'local_single' declared here}}

  // Assign multiple entities that are all __single
  local = p; // expected-note{{__single parameter 'p' assigned to 'local' here}}
  local = global_single_int; // expected-note{{__single global 'global_single_int' assigned to 'local' here}}
  local = local_single; // expected-note{{__single local variable 'local_single' assigned to 'local' here}}
  local = single_int_source(); // expected-note{{__single return value from call to 'single_int_source' assigned to 'local' here}}
  local = ptrs[0]; // expected-note{{__single element from array 'ptrs' assigned to 'local' here}}
  local = s->ptr; // expected-note{{__single struct member 'ptr' assigned to 'local' here}}
  local = u->ptr; // expected-note{{__single union member 'ptr' assigned to 'local' here}}
  local = sa->ptrs[1]; // expected-note{{__single struct member 'ptrs' assigned to 'local' here}}
  local = ua->ptrs[1]; // expected-note{{__single union member 'ptrs' assigned to 'local' here}}

  // expected-warning-re@+1{{indexing into __bidi_indexable local variable 'local' that is assigned from a __single pointer results in a trap{{$}}}}
  local[1] = 0;
}

//==============================================================================
// typedefs with VLAs
//==============================================================================

// FIXME(dliew): Clang currently warns twice because the size expression of the
// VLA type gets visited twice. rdar://117235333
void typedefs_with_vla(int *x, int size) { // expected-note 2{{pointer 'x' declared here}}
    int* local = x; // expected-note 2{{pointer 'local' initialized here}}
    for (int i = 0; i < size; ++i) {
        // expected-warning@+1 2{{indexing into a __bidi_indexable local variable 'local' initialized from __single parameter 'x' results in a trap if the index expression is >= 1 or < 0}}
        typedef int arr[local[i]];
        arr a; // has x[i] elements on each loop iteration
        if (i > 0)
            a[0] = 0;
    }
}

//==============================================================================
// False positives
//==============================================================================

void false_positive_reachability_ignored(int *p) { // expected-note{{pointer 'p' declared here}}
  int* local = p; // expected-note{{pointer 'local' initialized here}}

  if (0) {
    // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' initialized from __single parameter 'p' results in an out-of-bounds pointer}}
    ++local; // Not reachable
  }
}

void false_positive_reachability_ignored2(int *p) { // expected-note{{pointer 'p' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}

  if (0) {
    // expected-note@+1{{pointer 'local' assigned here}}
    local = p; // Not reachable
  }

  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in an out-of-bounds pointer}}
  ++local;
}

void false_positive_happens_before_ignored(int* p) { // expected-note{{pointer 'p' declared here}}
  int* local; // expected-note{{pointer 'local' declared here}}
  // expected-warning@+1{{pointer arithmetic over a __bidi_indexable local variable 'local' assigned from __single parameter 'p' results in an out-of-bounds pointer}}
  ++local;
  
  // The first __single assignment is made **after** the bad operation.
  // To fix this we need the analysis to understand happens-before (rdar://117166345).
  local = p; // expected-note{{pointer 'local' assigned here}}
}

void false_positive_assignment_notes(
  // expected-note@+1{{pointer 'p' declared here}}
  int* p,
  // expected-note@+1{{pointer 'q' declared here}}
  int* q,
  // expected-note@+1{{pointer 'r' declared here}}
  int* r) {
  // expected-note@+1{{pointer 'local' declared here}}
  int* local = p; // expected-note{{__single parameter 'p' assigned to 'local' here}}

  // This warning is not a false positive because `local` definitely has the
  // bounds of a `__single` by the time the unsafe operation happens.
  // expected-warning@+1{{pointer arithmetic over __bidi_indexable local variable 'local' that is assigned from a __single pointer results in an out-of-bounds pointer}}
  ++local;
  
  // These notes are false positives because these assignments happen **after** the unsafe.
  // operations. To fix this we need the analysis to understand happens-before (rdar://117166345).
  local = q; // expected-note{{__single parameter 'q' assigned to 'local' here}}
  local = r; // expected-note{{__single parameter 'r' assigned to 'local' here}}
}

//==============================================================================
// False negatives
//==============================================================================
void false_neg_multiple_assignment_different_expr(int* p, int* __bidi_indexable q) {
  // Multiple assignments from a different expr prevent a diagnostic from being
  // emitted.
  int* local = p;
  local = q;
  ++local; // Missing warning
}

void false_neg_happens_before_ignored(int* p, int* __bidi_indexable q) {
  // We fail to warn here because the analysis sees multiple definitions of
  // `local` but fails to take into account at `++local` only one definition
  // of local is possible. (rdar://117166345)
  int* local = p;
  ++local; // Missing warning
  local = q;
}

void false_neg_reachability_ignored(int* p, int* __bidi_indexable q) {
  // We fail to warn here because the analysis sees multiple definitions of
  // `local` but fails to take into account at `++local` only one definition
  // of local is possible.
  int* local = p;

  if (0) {
    local = q; // Not reachable
  }

  ++local; // Missing warning
}

void false_neg_transitive_assignments_ignored(int *p) {
  // We fail to warn here because the analysis doesn't track a __single
  // indirectly flowing into a local __bidi_indexable via another
  // local variable.
  int *local = p;
  int *local2 = local;
  ++local2; // Missing warning
}
