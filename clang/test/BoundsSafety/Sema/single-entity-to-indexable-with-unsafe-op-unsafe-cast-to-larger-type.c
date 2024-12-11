

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>

typedef int SmallTy;
typedef long long int BigTy;
_Static_assert(sizeof(BigTy) > sizeof(SmallTy), "expected size diff failed");


//==============================================================================
// Explicit Unsafe casts that stay as __bidi_indexable pointers
//==============================================================================
void unsafe_explicit_cast_assign_to_local(SmallTy* p) { // expected-note 2{{pointer 'p' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    BigTy* local2 = (BigTy*) local;

    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    local2 = (BigTy*) local;
}

void unsafe_explicit_cast_assign_to_local_multiple_singles(
    SmallTy* p, // expected-note 2{{pointer 'p' declared here}}
    SmallTy* q, // expected-note 2{{pointer 'q' declared here}}
    int condition) { 
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    if (condition)
        // expected-note@+1 2{{__single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
        local = q;
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    BigTy* local2 = (BigTy*) local;

    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    local2 = (BigTy*) local;
}

struct StructWithBigTyBidi {
    BigTy* __bidi_indexable field;
};


struct NestedStructWithBigTy {
    struct StructWithBigTyBidi nested;
    BigTy* __bidi_indexable field;
};

void unsafe_explicit_cast_struct_init(SmallTy* p) { // expected-note 3{{pointer 'p' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;

    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    struct StructWithBigTyBidi local2 = {.field = (BigTy*) local};

    struct NestedStructWithBigTy local3 = {
        // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
        .nested = {.field = (BigTy*) local},
        // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
        .field = (BigTy*) local};
}

BigTy* __bidi_indexable unsafe_explicit_return_bidi(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    return (BigTy*) local;
}

void receive_bidi(BigTy* __bidi_indexable);
void unsafe_explicit_cast_call_arg(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    receive_bidi((BigTy*) local);
}

void unsafe_explicit_cast_in_expr(SmallTy* p) { // expected-note 2{{pointer 'p' declared here}}
    // expected-note@+3 2{{pointer 'local' declared here}}
    // expected-note@+2{{_single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but '((BigTy *)local)' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+2{{indexing __bidi_indexable '((BigTy *)local)' at any index will access out-of-bounds memory and will trap in a future compiler version. At runtime 'local' is assigned a __single pointer that results in '((BigTy *)local)' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    ((BigTy*) local)[0] = 0;
}

typedef char TinyTy;
_Static_assert(sizeof(SmallTy) > sizeof(TinyTy), "expected size diff failed");

void unsafe_explicit_cast_assign_to_local_zero_elt_oob(TinyTy* p) { // expected-note 4{{pointer 'p' declared here}}
    // expected-note@+3 4{{pointer 'local' declared here}}
    // expected-note@+2 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but '(BigTy *)local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = (SmallTy* __bidi_indexable)(TinyTy* __bidi_indexable) p;
    // expected-warning@+2{{assigning from __bidi_indexable '(BigTy *)local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in '(BigTy *)local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    BigTy* local2 = (BigTy*) local;

    // expected-warning@+2{{assigning from __bidi_indexable '(BigTy *)local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in '(BigTy *)local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    local2 = (BigTy*) local;
}

void unsafe_explicit_cast_struct_init_zero_elt_oob(TinyTy* p) { // expected-note 6{{pointer 'p' declared here}}
    // expected-note@+3 6{{pointer 'local' declared here}}
    // expected-note@+2 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but '(BigTy *)local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = (SmallTy* __bidi_indexable)(TinyTy* __bidi_indexable) p;

    struct StructWithBigTyBidi local2 = {
        // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
        // expected-warning@+1{{assigning from __bidi_indexable '(BigTy *)local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in '(BigTy *)local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
        .field = (BigTy*) local
    };

    struct NestedStructWithBigTy local3 = {
        // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
        // expected-warning@+1{{assigning from __bidi_indexable '(BigTy *)local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in '(BigTy *)local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
        .nested = {.field = (BigTy*) local},
        // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
        // expected-warning@+1{{assigning from __bidi_indexable '(BigTy *)local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in '(BigTy *)local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
        .field = (BigTy*) local};
}

//==============================================================================
// Implicit Unsafe casts that stay as __bidi_indexable pointers
//==============================================================================
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
void unsafe_implicit_cast_assign_to_local(SmallTy* p) { // expected-note 2{{pointer 'p' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    BigTy* local2 = local;

    // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    local2 = local;
}

void unsafe_implicit_cast_struct_init(SmallTy* p) { // expected-note 3{{pointer 'p' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;

    // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    struct StructWithBigTyBidi local2 = {.field = local};

    struct NestedStructWithBigTy local3 = {
        // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
        .nested = {.field = local},
        // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
        .field = local};
}

BigTy* __bidi_indexable unsafe_implicit_return_bidi(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    return local;
}

void unsafe_implicit_cast_call_arg_bidi(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    receive_bidi(local);
}

void unsafe_implicit_cast_assign_to_local_zero_elt_oob(TinyTy* p) { // expected-note 4{{pointer 'p' declared here}}
    // expected-note@+3 4{{pointer 'local' declared here}}
    // expected-note@+2 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but cast of 'local' to 'BigTy *__bidi_indexable' (aka 'long long *__bidi_indexable') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but 'local' has pointee type 'SmallTy' (aka 'int') (4 bytes)}}
    SmallTy* local = (SmallTy* __bidi_indexable)(TinyTy* __bidi_indexable) p;
    // expected-warning@+2{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    BigTy* local2 = local;

    // expected-warning@+2{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{implicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    local2 = local;
}

#pragma clang diagnostic pop

//==============================================================================
// Explicit Unsafe casts that then are later converted to __single pointers
//==============================================================================
void unsafe_explicit_cast_assign_to_local_single(SmallTy* p) { // expected-note 2{{pointer 'p' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    BigTy* __single local2 = (BigTy*) local;

    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    local2 = (BigTy*) local;
}

struct StructWithBigTySingle {
    BigTy* field;
};


struct NestedStructWithBigTySingle {
    struct StructWithBigTySingle nested;
    BigTy* field;
};

void unsafe_explicit_cast_struct_init_single(SmallTy* p) { // expected-note 3{{pointer 'p' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;

    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    struct StructWithBigTySingle local2 = {.field = (BigTy*) local};

    struct NestedStructWithBigTySingle local3 = {
        // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
        .nested = {.field = (BigTy*) local},
        // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
        .field = (BigTy*) local};
}

BigTy* unsafe_explicit_return_single(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    return (BigTy*) local;
}

void receive_single(BigTy*);
void unsafe_explicit_cast_call_arg_single(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    receive_single((BigTy*) local);
}

void unsafe_explicit_cast_assign_to_local_zero_elt_oob_single(TinyTy* p) { // expected-note 4{{pointer 'p' declared here}}
    // expected-note@+3 4{{pointer 'local' declared here}}
    // expected-note@+2 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but '(BigTy *)local' has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = (SmallTy* __bidi_indexable)(TinyTy* __bidi_indexable) p;
    // expected-warning@+2{{assigning from __bidi_indexable '(BigTy *)local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in '(BigTy *)local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    BigTy* __single local2 = (BigTy*) local;

    // expected-warning@+2{{assigning from __bidi_indexable '(BigTy *)local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in '(BigTy *)local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    local2 = (BigTy*) local;
}

void unsafe_explicit_cast_in_expr_single(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{explicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    ((BigTy* __single) local)[0] = 5;
}

//==============================================================================
// Implicit unsafe casts that then are later converted to __single pointers
//==============================================================================
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
void unsafe_implicit_cast_assign_to_local_single(SmallTy* p) { // expected-note 2{{pointer 'p' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    BigTy* __single local2 = local;

    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    local2 = local;
}

void unsafe_implicit_cast_struct_init_single(SmallTy* p) { // expected-note 3{{pointer 'p' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;

    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    struct StructWithBigTySingle local2 = {.field = local};

    struct NestedStructWithBigTySingle local3 = {
        // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
        .nested = {.field = local},
        // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
        .field = local};
}

BigTy* unsafe_implicit_return_single(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    return local;
}

void unsafe_implicit_cast_call_arg_single(SmallTy* p) { // expected-note{{pointer 'p' declared here}}
    // expected-note@+2{{pointer 'local' declared here}}
    // expected-note@+1{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'SmallTy' (aka 'int') (4 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    SmallTy* local = p;
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    receive_single(local);
}

void unsafe_implicit_cast_assign_to_local_zero_elt_oob_single(TinyTy* p) { // expected-note 4{{pointer 'p' declared here}}
    // expected-note@+3 4{{pointer 'local' declared here}}
    // expected-note@+2 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but cast of 'local' to 'BigTy *__single' (aka 'long long *__single') has pointee type 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-note@+1 2{{__single parameter 'p' used to initialize 'local' here results in 'local' having the bounds of a single 'TinyTy' (aka 'char') (1 bytes) but 'local' has pointee type 'SmallTy' (aka 'int') (4 bytes)}}
    SmallTy* local = (SmallTy* __bidi_indexable)(TinyTy* __bidi_indexable) p;
    // expected-warning@+2{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    BigTy* __single local2 = local;

    // expected-warning@+2{{assigning from __bidi_indexable 'local' will propagate an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'BigTy' (aka 'long long') (8 bytes)}}
    // expected-warning@+1{{implicit cast of out-of-bounds __bidi_indexable to __single will trap in a future compiler version due to the bounds of 'local' being too small to access a single element of type 'BigTy' (aka 'long long')}}
    local2 = local;
}

#pragma clang diagnostic pop
