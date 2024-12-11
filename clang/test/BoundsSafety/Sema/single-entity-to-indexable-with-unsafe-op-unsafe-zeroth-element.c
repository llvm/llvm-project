

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>

typedef int SmallTy;
typedef long long int BigTy;
_Static_assert(sizeof(BigTy) > sizeof(SmallTy), "expected size diff failed");

//==============================================================================
// Local __bidi_indexable with bounds smaller than its element type (i.e. the
// 0th element can't be accessed safely).
//
// Currently we don't generate traps for this (rdar://119744147) but we still
// warn because it will become a trap in the future.
//==============================================================================

void param_single_explicit_cast_deref(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
  // expected-note@+2{{pointer 'local' declared here}}
  // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  BigTy* local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) p;
  // expected-warning@+1{{dereferencing __bidi_indexable 'local' will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  *local = 0;
}

void param_single_explicit_cast_deref_with_cast_to_larger(SmallTy* p) { // expected-note 2{{pointer 'p' declared here}}
  // expected-note@+3 2{{pointer 'local' declared here}}
  // expected-note@+2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but '((BigTy *)local)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  SmallTy* local = p;
  // expected-warning@+2{{dereferencing __bidi_indexable '((BigTy *)local)' will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local' is assigned a __single pointer that results in '((BigTy *)local)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
  *((BigTy*)local) = 0;
}


void param_single_explicit_cast_index(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
  // expected-note@+2{{pointer 'local' declared here}}
  // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  BigTy* local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) p;
  // expected-warning@+1{{indexing __bidi_indexable 'local' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
  local[0] = 0;
}

struct SmallStructTy {
    int a;
    int b;
};
_Static_assert(sizeof(struct SmallStructTy) == 8, "wrong size");

struct BigStructTy {
    struct SmallStructTy inner;
    int c;
};
_Static_assert(sizeof(struct BigStructTy) == 12, "wrong size");
_Static_assert(sizeof(struct SmallStructTy) < sizeof(struct BigStructTy), "expected size diff failed");

void param_single_explicit_cast_struct_deref(struct SmallStructTy* p, struct BigStructTy* q) { // expected-note{{pointer 'p' declared here}}
  // expected-note@+2{{pointer 'local' declared here}}
  // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but 'local' has pointee type 'struct BigStructTy' (12 bytes)}}
  struct BigStructTy* local = (struct BigStructTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;
  // expected-warning@+1{{dereferencing __bidi_indexable 'local' will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructTy'}}
  *local = *q;
}

struct StructWithBigTyBidi {
    BigTy* __bidi_indexable member;
};

void receiveBigTyBidi(BigTy* __bidi_indexable);
BigTy* __bidi_indexable param_single_explicit_cast_oob_escapes_bidi(SmallTy* p) { // expected-note 4{{pointer 'p' declared here}}
  // expected-note@+2 4{{pointer 'local' declared here}}
  // expected-note@+1 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  BigTy* local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) p;

  // expected-warning@+1{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  BigTy* local2 = local;

  // expected-warning@+1{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  struct StructWithBigTyBidi local3 = { .member = local };

  // expected-warning@+1{{passing __bidi_indexable 'local' will pass an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  receiveBigTyBidi(local);

  // expected-warning@+1{{returning __bidi_indexable 'local' will return an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  return local;
}

// Test that multiple single 'entities' are supported. This is in no way
// exhaustive.

SmallTy* retSmallTy(void); // expected-note 4{{'retSmallTy' declared here}}
struct StructWithSmallTySingle {
    SmallTy*  member; // expected-note 4{{StructWithSmallTySingle::member declared here}}
};
SmallTy* globalSmallTy; // expected-note 4{{pointer 'globalSmallTy' declared here}}
BigTy* __bidi_indexable param_single_explicit_cast_oob_escapes_bidi_multiple_singles(
  SmallTy* p, // expected-note 4{{pointer 'p' declared here}}
  int condition,
  struct StructWithSmallTySingle s) {
  SmallTy* arr[2]; // expected-note 4{{'arr' declared here}}
  // expected-note@+1 4{{pointer 'local' declared here}}
  BigTy* local;
  if (condition == 0)
    // expected-note@+1 4{{__single parameter 'p' assigned to 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) p;
  else if (condition == 1)
    // expected-note@+1 4{{__single global 'globalSmallTy' assigned to 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) globalSmallTy;
  else if (condition == 2)
    // expected-note@+1 4{{__single return value from call to 'retSmallTy' assigned to 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) retSmallTy();
  else if (condition == 3)
    // expected-note@+1 4{{__single struct member 'member' assigned to 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) s.member;
  else if (condition == 4)
    // expected-note@+1 4{{_single element from array 'arr' assigned to 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) arr[0];

  // expected-warning@+1{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  BigTy* local2 = local;

  // expected-warning@+1{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  struct StructWithBigTyBidi local3 = { .member = local };

  // expected-warning@+1{{passing __bidi_indexable 'local' will pass an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  receiveBigTyBidi(local);

  // expected-warning@+1{{returning __bidi_indexable 'local' will return an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  return local;
}

struct StructWithBigTySingle {
    BigTy*  member;
};

void receiveBigTySingle(BigTy*);
BigTy* param_single_explicit_cast_oob_escapes_single(SmallTy* p) { // expected-note 8{{pointer 'p' declared here}}
  // expected-note@+3 8{{pointer 'local' declared here}}
  // expected-note@+2 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  // expected-note@+1 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but 'local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
  BigTy* local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) p;

  // expected-warning@+2{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
  // expected-warning@+1{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  BigTy* __single local2 = local;

  // expected-warning@+2{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
  // expected-warning@+1{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  struct StructWithBigTySingle local3 = { .member = local };

  // expected-warning@+2{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
  // expected-warning@+1{{passing __bidi_indexable 'local' will pass an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  receiveBigTySingle(local);

  // expected-warning@+2{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
  // expected-warning@+1{{returning __bidi_indexable 'local' will return an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long')}}
  return local;
}

// TODO: add __single variant

//==============================================================================
// Member access
//
// Despite the bounds of the local (that suggest some part of it can
// be accessed) access through `->` will trap if **any** field is accessed
// due to `CodeGenFunction::EmitMemberExpr` taking the size of the base expr
// to `->` into account.
//==============================================================================

void param_single_explicit_cast_nested_member_access(struct SmallStructTy* p) { // expected-note 3{{pointer 'p' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but 'local' has pointee type 'struct BigStructTy' (12 bytes)}}
    struct BigStructTy* local = (struct BigStructTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;

    // The bounds of `local` suggest that `inner` is accessible but
    // `CodeGenFunction::EmitMemberExpr` takes the struct size into account so
    // no part of the struct is accessible through `->`.
    //
    // expected-warning@+1{{accessing field BigStructTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructTy'}}
    local->inner.a = 0;
    // expected-warning@+1{{accessing field BigStructTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructTy'}}
    local->inner.b = 0;

    // expected-warning@+1{{accessing field BigStructTy::c through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructTy'}}
    local->c = 0;
}

void param_single_nested_member_access_explicit_cast_to_larger_at_use(struct SmallStructTy* p) { // expected-note 6{{pointer 'p' declared here}}
    // expected-note@+3 6{{pointer 'local' declared here}}
    // expected-note@+2 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but cast of 'local' to 'struct BigStructTy *__bidi_indexable' has pointee type 'struct BigStructTy' (12 bytes)}}
    // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but '((struct BigStructTy *)local)' has pointee type 'struct BigStructTy' (12 bytes}}
    struct SmallStructTy* local = p;

    // The bounds of `local` suggest that `inner` is accessible but
    // `CodeGenFunction::EmitMemberExpr` takes the struct size into account so
    // no part of the struct is accessible through `((BigStructTy*)local)->`
    //
    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may tra}}
    // expected-warning@+1{{accessing field BigStructTy::inner through __bidi_indexable '((struct BigStructTy *)local)' will always trap. At runtime 'local' is assigned a __single pointer that results in '((struct BigStructTy *)local)' having bounds smaller than a single 'struct BigStructTy' (12 bytes)}}
    ((struct BigStructTy*) local)->inner.a = 0;

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may tra}}
    // expected-warning@+1{{accessing field BigStructTy::inner through __bidi_indexable '((struct BigStructTy *)local)' will always trap. At runtime 'local' is assigned a __single pointer that results in '((struct BigStructTy *)local)' having bounds smaller than a single 'struct BigStructTy' (12 bytes)}}
    ((struct BigStructTy*) local)->inner.b = 0;

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may tra}}
    // expected-warning@+1{{accessing field BigStructTy::c through __bidi_indexable '((struct BigStructTy *)local)' will always trap. At runtime 'local' is assigned a __single pointer that results in '((struct BigStructTy *)local)' having bounds smaller than a single 'struct BigStructTy' (12 bytes)}}
    ((struct BigStructTy*) local)->c = 0;
}

void param_single_nested_member_access_explicit_cast_to_smaller_at_use(struct BigStructTy* p) {
    struct BigStructTy* local = p;

    // No warnings when casting to a smaller type.
    ((struct SmallStructTy*) local)->a = 0;
    ((struct SmallStructTy*) local)->b = 0;
}

struct BigStructWithFAMTy {
    struct SmallStructTy inner;
    int c;
    char buf[__counted_by(c)];
};

void param_single_explicit_cast_fam_member_access(struct SmallStructTy* p) { // expected-note 5{{pointer 'p' declared here}}
    // expected-note@+2 5{{pointer 'local' declared here}}
    // expected-note@+1 5{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but 'local' has pointee type 'struct BigStructWithFAMTy' (12 bytes)}}
    struct BigStructWithFAMTy* local = (struct BigStructWithFAMTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;

    // The bounds of `local` suggest that `inner` is accessible but
    // `CodeGenFunction::EmitMemberExpr` takes the struct size into account so
    // no part of the struct is accessible through `->`.
    //
    // expected-warning@+1{{accessing field BigStructWithFAMTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithFAMTy'}}
    local->inner.a = 0;
    // expected-warning@+1{{accessing field BigStructWithFAMTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithFAMTy'}}
    local->inner.b = 0;

    // expected-warning@+1{{accessing field BigStructWithFAMTy::buf through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithFAMTy'}}
    local->buf[0] = 0;

    // expected-warning@+1{{accessing field BigStructWithFAMTy::buf through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithFAMTy'}}
    char* local2 = local->buf + 1;

    // expected-warning@+1{{accessing field BigStructWithFAMTy::buf through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithFAMTy'}}
    char* local3 = &(local->buf[0]);
}

struct BigStructWithBitFieldsTy {
    struct SmallStructTy inner;
    int c:1;
    int d:1;
};

void param_single_explicit_cast_bit_field_access(struct SmallStructTy* p) { // expected-note 4{{pointer 'p' declared here}}
    // expected-note@+2 4{{pointer 'local' declared here}}
    // expected-note@+1 4{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but 'local' has pointee type 'struct BigStructWithBitFieldsTy' (12 bytes)}}
    struct BigStructWithBitFieldsTy* local = (struct BigStructWithBitFieldsTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;
   
    // The bounds of `local` suggest that `inner` is accessible but
    // `CodeGenFunction::EmitMemberExpr` takes the struct size into account so
    // no part of the struct is accessible through `->`.
    //
    // expected-warning@+1{{accessing field BigStructWithBitFieldsTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithBitFieldsTy'}}
    local->inner.a = 0;
    // expected-warning@+1{{accessing field BigStructWithBitFieldsTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithBitFieldsTy'}}
    local->inner.b = 0;

    // expected-warning@+1{{accessing field BigStructWithBitFieldsTy::c through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithBitFieldsTy'}}
    local->c = 0;

    // expected-warning@+1{{accessing field BigStructWithBitFieldsTy::d through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithBitFieldsTy'}}
    local->d = 0;
}

struct BigStructWithUnannotatedFAMTy {
    struct SmallStructTy inner;
    int c;
    char buffer[];
};

void param_single_explicit_cast_unannotated_fam_member_access(struct SmallStructTy* p) { // expected-note 2{{pointer 'p' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but 'local' has pointee type 'struct BigStructWithUnannotatedFAMTy' (12 bytes)}}
    struct BigStructWithUnannotatedFAMTy* local = (struct BigStructWithUnannotatedFAMTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;

    // The bounds of `local` suggest that `inner` is accessible but
    // `CodeGenFunction::EmitMemberExpr` takes the struct size into account so
    // no part of the struct is accessible through `->`.
    //
    // expected-warning@+1{{accessing field BigStructWithUnannotatedFAMTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithUnannotatedFAMTy'}}
    local->inner.a = 0;
    // expected-warning@+1{{accessing field BigStructWithUnannotatedFAMTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'struct BigStructWithUnannotatedFAMTy'}}
    local->inner.b = 0;

    // The warning is suppressed in this case because `warn_bounds_safety_promoting_incomplete_array_without_count` already fires.
    // expected-warning@+1{{accessing elements of an unannotated incomplete array always fails at runtime}}
    local->buffer[0] = 0;
}

union BigUnionTy {
    struct SmallStructTy inner;
    struct BigStructWithFAMTy big_inner;
};

void param_single_explicit_cast_union_access(struct SmallStructTy* p) { // expected-note 6{{pointer 'p' declared here}}
    // expected-note@+2 6{{pointer 'local' declared here}}
    // expected-note@+1 6{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'struct SmallStructTy' (8 bytes) but 'local' has pointee type 'union BigUnionTy' (12 bytes)}}
    union BigUnionTy* local = (union BigUnionTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;

    // expected-warning@+1{{accessing field BigUnionTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'union BigUnionTy'}}
    local->inner.a = 0;
    // expected-warning@+1{{accessing field BigUnionTy::inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'union BigUnionTy'}}
    local->inner.b = 0;
    // expected-warning@+1{{accessing field BigUnionTy::big_inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'union BigUnionTy'}}
    local->big_inner.inner.a = 0;
    // expected-warning@+1{{accessing field BigUnionTy::big_inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'union BigUnionTy'}}
    local->big_inner.inner.b = 0;

    // expected-warning@+1{{accessing field BigUnionTy::big_inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'union BigUnionTy'}}
    local->big_inner.c = 0;
    // expected-warning@+1{{accessing field BigUnionTy::big_inner through __bidi_indexable 'local' will always trap. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'union BigUnionTy'}}
    local->big_inner.buf[0] = 0;
}

//==============================================================================
// False negatives
//==============================================================================

void param_single_explicit_cast_not_all_assignments_too_small(SmallTy* p) {
  BigTy* local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) p;
  *local = 0; // No warn

  // Currently happens-before isn't taken into account so this assignment is
  // considered which means not all assignments to `local` are a __single pointer
  // that has element type which is too small.
  local = 0;
}

void param_single_explicit_cast_not_all_assignments_too_small2(SmallTy* p, BigTy* q, int condition) {
  BigTy* local;
  if (condition)
    local = (BigTy* __bidi_indexable)(SmallTy* __bidi_indexable) p;
  else
   local = q;

  *local = 0; // No warn
}

struct OtherStructWithAnnotatedFAMTy {
    int a;
    int b;
    char buffer[__counted_by(a)];
};
_Static_assert(sizeof(struct OtherStructWithAnnotatedFAMTy) == sizeof(struct SmallStructTy), "size mismatch");

void param_single_explicit_cast_annotated_fam_member_access_false_neg(struct SmallStructTy* p) {
    struct OtherStructWithAnnotatedFAMTy* local = (struct OtherStructWithAnnotatedFAMTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;
    local->a = 0;
    local->b = 0;

    // TODO: We should warn about this because its a guaranteed runtime trap.
    // rdar://120566596
    local->buffer[0] = 0;
}

struct OtherStructWithUnannotatedFAMTy {
    int a;
    int b;
    char buffer[];
};
_Static_assert(sizeof(struct OtherStructWithUnannotatedFAMTy) == sizeof(struct SmallStructTy), "size mismatch");

void param_single_explicit_cast_unannotated_fam_member_access_false_neg(struct SmallStructTy* p) {
    struct OtherStructWithUnannotatedFAMTy* local = (struct OtherStructWithUnannotatedFAMTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;
    local->a = 0;
    local->b = 0;

    // This existing warning already fires so this case doesn't need additional
    // support but this test case is here to make sure this doesn't regress.
    // expected-warning@+1{{accessing elements of an unannotated incomplete array always fails at runtime}}
    local->buffer[0] = 0;
}

//==============================================================================
// No warnings expected
//==============================================================================

// Same layout as `SmallStructTy` but not the same type
struct OtherSmallStructTy {
    int a;
    int b;
};
_Static_assert(sizeof(struct OtherSmallStructTy) == sizeof(struct SmallStructTy), "size mismatch");

void param_single_explicit_cast_same_struct_size(struct SmallStructTy* p, struct OtherSmallStructTy* q) {
  struct OtherSmallStructTy* local = (struct OtherSmallStructTy* __bidi_indexable)(struct SmallStructTy* __bidi_indexable) p;
  *local = *q;
}

// Flexible Array Members

struct SmallFAM {
    int count;
    char buf[__counted_by(count)];
};
_Static_assert(sizeof(struct SmallFAM) == 4, "wrong size");

struct BiggerFAM {
    int count;
    int extra_field;
    char buf[__counted_by(count)];
};
_Static_assert(sizeof(struct BiggerFAM) == 8, "wrong size");

_Static_assert(sizeof(struct SmallFAM) < sizeof(struct BiggerFAM), "expected size diff failed");

void param_single_explicit_cast_fam_struct_deref(struct SmallFAM* p) {
    struct BiggerFAM* local = (struct BiggerFAM* __bidi_indexable)(struct SmallFAM* __bidi_indexable) p;
    // No warn
    // Even though `p` is a `__single` when it is assigned to `local` the bounds
    // are dynamically computed from `p`'s count field. Thus the compiler can't
    // know what the bounds of `local` will be at runtime.
    local[0].count = 0;
}
