
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>
#include <stddef.h>

//==============================================================================
// Constant 0 count
//==============================================================================

// not a trap
void cb_const_size_zero(int *__counted_by(0) buf) {}
int *__counted_by(0) use_cb_const_size_zero(int *buf) {
    int *local = buf;
    cb_const_size_zero(local);

    int *__counted_by(0) local2 = local;
    return local;
}

// not a trap
void cbon_const_size_zero(int *__counted_by_or_null(0) buf) {}
int *__counted_by_or_null(0) use_cbon_const_size_zero(int *buf) {
    int *local = buf;
    cbon_const_size_zero(local);

    int *__counted_by_or_null(0) local2 = local;
    return local;
}

// not a trap
void sb_const_size_zero(void *__sized_by(0) buf) {}
void *__sized_by(0) use_sb_const_size_zero(int *buf) {
    int *local = buf;
    sb_const_size_zero(local);

    void *__sized_by(0) local2 = local;
    return local;
}

// not a trap
void sbon_const_size_zero(void *__sized_by_or_null(0) buf) {}
void *__sized_by_or_null(0) use_sbon_const_size_zero(int *buf) {
    int *local = buf;
    sbon_const_size_zero(local);

    void *__sized_by_or_null(0) local2 = local;
    return local;
}

//==============================================================================
// __sized_by constant count sizeof(int)
//==============================================================================

// not a trap (provided buf != null)
void sb_const_sizeof_int(void *__sized_by(sizeof(int)) buf) {}
void *__sized_by(sizeof(int)) use_sb_const_sizeof_int(int *buf) {
    int *local = buf;
    sb_const_sizeof_int(local);

    void *__sized_by(sizeof(int)) local2 = local;
    return local;
}

// trap - false negative warning
void *__sized_by(sizeof(int)) use_sb_const_sizeof_int_null_init(int *buf) {
    int *local = 0;
    sb_const_sizeof_int(local);

    void *__sized_by(sizeof(int)) local2 = local;
    return local;
}


// not a trap
void sbon_const_sizeof_int(void *__sized_by_or_null(sizeof(int)) buf) {}
void *__sized_by_or_null(sizeof(int)) use_sbon_const_sizeof_int(int *buf) {
    int *local = buf;
    sbon_const_sizeof_int(local);

    void *__sized_by_or_null(sizeof(int)) local2 = local;
    return local;
}

//==============================================================================
// __sized_by constant count sizeof(int)+1
//==============================================================================

// trap
void sb_const_sizeof_int_plus_one(void *__sized_by(sizeof(int)+1) buf) {}
void *__sized_by(sizeof(int)+1) use_sb_const_sizeof_int_plus_one(int *buf) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'void *__single __sized_by(5UL)' (aka 'void *__single') requires 5 bytes or more}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap when converting to 'void *__single __sized_by(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    sb_const_sizeof_int_plus_one(local);

    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap when converting to 'void *__single __sized_by(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    void *__sized_by(sizeof(int)+1) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will trap in a future compiler version when converting to 'void *__single __sized_by(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    return local;
}



// trap if local != null
void sbon_const_sizeof_int_plus_one(void *__sized_by_or_null(sizeof(int)+1) buf) {}
void *__sized_by_or_null(sizeof(int)+1) use_sbon_const_sizeof_int_plus_one(int *buf) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'void *__single __sized_by_or_null(5UL)' (aka 'void *__single') requires 5 bytes or more}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'void *__single __sized_by_or_null(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    sbon_const_sizeof_int_plus_one(local);

    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'void *__single __sized_by_or_null(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    void *__sized_by_or_null(sizeof(int)+1) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will trap (unless 'local' is null) in a future compiler version when converting to 'void *__single __sized_by_or_null(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    return local;
}

// not a trap
void use_sbon_const_sizeof_int_plus_one_null_ptr_reach_call(int *buf) {
    int *local = (int* __single) 0;
    sbon_const_sizeof_int_plus_one(local); // At callsite we know the pointer is null.
}



//==============================================================================
// __counted_by constant count 1
//==============================================================================

// not a trap (provided local != null)
void cb_const_size_one(int *__counted_by(1) buf) {}
int *__counted_by(1) use_cb_const_size_one(int* buf) {
    int *local = buf;
    cb_const_size_one(local);

    int *__counted_by(1) local2 = local;
    return local;
}



// not a trap
void cbon_const_size_one(int *__counted_by_or_null(1) buf) {}
int *__counted_by_or_null(1) use_cbon_const_size_one(int* buf) {
    int *local = buf;
    cbon_const_size_one(local);

    int *__counted_by_or_null(1) local2 = local;
    return local;
}


//==============================================================================
// __counted_by constant count > 1
//==============================================================================

// trap
void cb_const_size_two(int *__counted_by(2) buf) {}
int *__counted_by(2) use_cb_const_size_two(int* buf) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'int *__single __counted_by(2)' (aka 'int *__single') requires 8 bytes or more}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap when converting to 'int *__single __counted_by(2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    cb_const_size_two(local);

    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap when converting to 'int *__single __counted_by(2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    int *__counted_by(2) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will trap in a future compiler version when converting to 'int *__single __counted_by(2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    return local;
}


// trap if ptr != null
void cbon_const_size_two(int *__counted_by_or_null(2) buf) {}
int *__counted_by_or_null(2) use_cbon_const_size_two(int* buf) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'int *__single __counted_by_or_null(2)' (aka 'int *__single') requires 8 bytes or more}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'int *__single __counted_by_or_null(2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    cbon_const_size_two(local);

    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'int *__single __counted_by_or_null(2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    int *__counted_by_or_null(2) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will trap (unless 'local' is null) in a future compiler version when converting to 'int *__single __counted_by_or_null(2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    return local;
}



// trap
_Static_assert(sizeof(long long int) > sizeof(int), "unexpected diff");
void cb_const_size_two_ll(long long int *__counted_by(2) buf) {}
long long int *__counted_by(2) use_cb_const_size_two_ll(int* buf) { // expected-note 6{{pointer 'buf' declared here}}
    // This tests the pointee size types (int vs long long int) not matching.

    // expected-note@+3 6{{pointer 'local' declared here}}
    // expected-note@+2 3{{__single parameter 'buf' used to initialize 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but cast of 'local' to 'long long *__bidi_indexable' has pointee type 'long long' (8 bytes)}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'long long *__single __counted_by(2)' (aka 'long long *__single') requires 16 bytes or more}}
    int *local = buf;
    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap when converting to 'long long *__single __counted_by(2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    cb_const_size_two_ll((long long int*)local);

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap when converting to 'long long *__single __counted_by(2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    long long int *__counted_by(2) local2 = (long long int*) local;

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will trap in a future compiler version when converting to 'long long *__single __counted_by(2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    return (long long int*)local;
}

void cb_const_size_one_ll(long long int *__counted_by(1) buf) {}
long long int *__counted_by(1) use_cb_const_size_one_ll(int* buf) { // expected-note 6{{pointer 'buf' declared here}}
    // This tests the pointee size types (int vs long long int) not matching.
    // expected-note@+3 6{{pointer 'local' declared here}}
    // expected-note@+2 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'long long *__single __counted_by(1)' (aka 'long long *__single') requires 8 bytes or more}}
    // expected-note@+1 3{{__single parameter 'buf' used to initialize 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but cast of 'local' to 'long long *__bidi_indexable' has pointee type 'long long' (8 bytes)}}
    int *local = buf;

    // expected-warning@+2{{passing __bidi_indexable local variable 'local' will trap when converting to 'long long *__single __counted_by(1)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    cb_const_size_one_ll((long long int*) local);

    // expected-warning@+2{{assigning from __bidi_indexable local variable 'local' will trap when converting to 'long long *__single __counted_by(1)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    long long int *__counted_by(1) local2 = (long long int*) local;

    // expected-warning@+2{{returning __bidi_indexable local variable 'local' will trap in a future compiler version when converting to 'long long *__single __counted_by(1)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    return (long long int*) local;
}


// trap if ptr != null
void cbon_const_size_two_ll(long long int *__counted_by_or_null(2) buf) {}
long long int *__counted_by_or_null(2) use_cbon_const_size_two_ll(int* buf) { // expected-note 6{{pointer 'buf' declared here}}
    // This tests the pointee size types (int vs long long int) not matching.

    // expected-note@+3 6{{pointer 'local' declared here}}
    // expected-note@+2 3{{__single parameter 'buf' used to initialize 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but cast of 'local' to 'long long *__bidi_indexable' has pointee type 'long long' (8 bytes)}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'long long *__single __counted_by_or_null(2)' (aka 'long long *__single') requires 16 bytes or more}}
    int *local = buf;
    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'long long *__single __counted_by_or_null(2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    cbon_const_size_two_ll((long long int*)local);

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'long long *__single __counted_by_or_null(2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    long long int *__counted_by_or_null(2) local2 = (long long int*) local;

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will trap (unless 'local' is null) in a future compiler version when converting to 'long long *__single __counted_by_or_null(2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer}}
    return (long long int*)local;
}

//==============================================================================
// Non const size in type
// __counted_by/__counted_by_or_null
//==============================================================================

void cb_non_const_size(int *__counted_by(size) buf, size_t size) {}
void cbon_non_const_size(int *__counted_by_or_null(size) buf, size_t size) {}


// - if ptr != null then traps if size != 1 and size != 0
// - if ptr null then traps if size != 0
int *__counted_by(size) use_cb_non_const_size_non_const_arg(int *buf, size_t size) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by(size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    cb_non_const_size(local, size);

    size_t size2 = size;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    int *__counted_by(size2) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by(size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    return local;
}


// if ptr null, never traps
// if ptr != null then traps if size != 1 and size != 0
int *__counted_by_or_null(size) use_cbon_non_const_size_non_const_arg(int *buf, size_t size) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by_or_null(size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    cbon_non_const_size(local, size);

    size_t size2 = size;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by_or_null(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    int *__counted_by_or_null(size2) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by_or_null(size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    return local;
}

// not a trap for call
// maybe a trap for return
int *__counted_by(*size) use_cb_non_const_size_zero_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // not a trap because the analysis knows that `size` is 0.
    cb_non_const_size(local, 0);

    // False positive
    // The analysis doesn't know that `size2` is 0
    size_t size2 = 0;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    int *__counted_by(size2) local2 = local;

    // not a trap because the analysis knows that `size` is 0.
    const size_t size3 = 0;
    int *__counted_by(size3) local3 = local;

    // False positive
    // The analysis doesn't know that `*size` is 0
    *size = 0;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by(*size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    return local;
}

// not a trap
int *__counted_by_or_null(*size) use_cbon_non_const_size_zero_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // not a trap because the analysis knows that `size` is 0.
    cbon_non_const_size(local, 0);

    // False positive
    // The analysis doesn't know that `size2` is 0
    size_t size2 = 0;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by_or_null(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    int *__counted_by_or_null(size2) local2 = local;

    // not a trap because the analysis knows that `size` is 0.
    const size_t size3 = 0;
    int *__counted_by_or_null(size3) local3 = local;

    // False positive
    // The analysis doesn't know that `*size` is 0
    *size = 0;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by_or_null(*size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    return local;
}

// traps iff ptr == null
int *__counted_by(*size) use_cb_non_const_size_one_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // not a trap in most cases because the analysis knows that `size` is 1.
    // Currently doesn't warn about the trap when local == null.
    cb_non_const_size(local, 1);

    // False positive
    // The analysis doesn't know that `size2` is 1
    size_t size2 = 1;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    int *__counted_by(size2) local2 = local;

    // not a trap because the analysis knows that `size` is 1.
    const size_t size3 = 1;
    int *__counted_by(size3) local3 = local;

    // False positive
    // The analysis doesn't know that `*size` is 0
    *size = 0;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by(*size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    return local;
}

// not a trap
int* __counted_by_or_null(*size) use_cbon_non_const_size_one_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // not a trap because the analysis knows that `size` is 1.
    cbon_non_const_size(local, 1);

    // False positive
    // The analysis doesn't know that `size2` is 1
    size_t size2 = 1;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by_or_null(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    int *__counted_by_or_null(size2) local2 = local;

    // not a trap because the analysis knows that `size` is 1.
    const size_t size3 = 1;
    int *__counted_by_or_null(size3) local3 = local;

    // False positive
    // The analysis doesn't know that `*size` is 1
    *size = 1;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by_or_null(*size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    return local;
}

// trap
int* __counted_by(*size) use_cb_non_const_size_two_arg(int *buf, size_t *size) { // expected-note 4{{pointer 'buf' declared here}}
    // expected-note@+4 4{{pointer 'local' declared here}}
    // expected-note@+3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'int *__single __counted_by(size)' (aka 'int *__single') requires 8 bytes or more}}
    // expected-note@+2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'int *__single __counted_by(2UL)' (aka 'int *__single') requires 8 bytes or more}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap when converting to 'int *__single __counted_by(size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    cb_non_const_size(local, 2);

    // The analysis isn't aware that `size2` is 2 so the warning is not as
    // specific as it should be.
    size_t size2 = 2;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    int *__counted_by(size2) local2 = local;

    // The analysis is aware that `size` is 2 so the warning is more specific
    const size_t size3 = 2;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap when converting to 'int *__single __counted_by(2UL)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    int *__counted_by(size3) local3 = local;

    // The analysis isn't aware that `*size` is 2 so the warning is not as
    // specific as it should be.
    *size = 2;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by(*size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    return local;
}

// trap if ptr != null
int* __counted_by_or_null(*size) use_cbon_non_const_size_two_arg(int *buf, size_t *size) { // expected-note 4{{pointer 'buf' declared here}}
    // expected-note@+4 4{{pointer 'local' declared here}}
    // expected-note@+3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'int *__single __counted_by_or_null(size)' (aka 'int *__single') requires 8 bytes or more}}
    // expected-note@+2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'int *__single __counted_by_or_null(2UL)' (aka 'int *__single') requires 8 bytes or more}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'int *__single __counted_by_or_null(size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    cbon_non_const_size(local, 2);

    // The analysis isn't aware that `size` is 2 so the warning is not as
    // specific as it should be.
    size_t size2 = 2;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by_or_null(size2)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    int *__counted_by_or_null(size2) local2 = local;

    // The analysis is aware that `size` is 2 so the warning is more specific
    const size_t size3 = 2;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'int *__single __counted_by_or_null(2UL)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer}}
    int *__counted_by_or_null(size3) local3 = local;

    // The analysis isn't aware that `*size` is 2 so the warning not as specific
    // as it should be.
    *size = 2;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'int *__single __counted_by_or_null(*size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap}}
    return local;
}

// - if ptr != null then traps if size > 0
// - if ptr null then traps if size > 0
void cb_non_const_size_ll(long long int* __counted_by(size) local, size_t size);
long long int *__counted_by(size) use_cb_non_const_size_non_const_arg_ll(int *buf, size_t size) { // expected-note 7{{pointer 'buf' declared here}}
    // This tests the pointee size types (sizeof(int) < sizeof(long long int)) not matching.

    // expected-note@+3 7{{pointer 'local' declared here}}
    // expected-note@+2 4{{__single parameter 'buf' used to initialize 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but cast of 'local' to 'long long *__bidi_indexable' has pointee type 'long long' (8 bytes)}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'long long *__single __counted_by(size)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 0 will trap. If 'local' is null then any count != 0 will trap}}
    cb_non_const_size_ll((long long int*) local, size);

    // FIXME: Consider suppressing the cast warning if it flows into something
    // we know won't trap.
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    cb_non_const_size_ll((long long int*) local, 0); // not a trap

    size_t size2 = size;
    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'long long *__single __counted_by(size2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 0 will trap. If 'local' is null then any count != 0 will trap}}
    long long int *__counted_by(size2) local2 = (long long int*) local;

    // expected-warning@+2{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'long long *__single __counted_by(size)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 0 will trap. If 'local' is null then any count != 0 will trap}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    return (long long int*) local;
}

void cb_non_const_size_c(char* __counted_by(size) local, size_t size);
char *__counted_by(size) use_cb_non_const_size_non_const_arg_c(int *buf, size_t size) { // expected-note 3{{pointer 'buf' declared here}}
    // This tests the pointee size types (sizeof(int) > sizeof(char)) not matching.

    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'char *__single __counted_by(size)' (aka 'char *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 4 will trap. If 'local' is null then any count != 0 will trap}}
    cb_non_const_size_c((char*) local, size);

    cb_non_const_size_c((char*) local, 4); // not a trap
    cb_non_const_size_c((char*) local, 3); // not a trap
    cb_non_const_size_c((char*) local, 2); // not a trap
    cb_non_const_size_c((char*) local, 1); // not a trap

    size_t size2 = size;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'char *__single __counted_by(size2)' (aka 'char *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 4 will trap. If 'local' is null then any count != 0 will trap}}
    char *__counted_by(size2) local2 = (char*) local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'char *__single __counted_by(size)' (aka 'char *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 4 will trap. If 'local' is null then any count != 0 will trap}}
    return (char*) local;
}

// - if ptr != null then traps if size > 0
void cbon_non_const_size_ll(long long int* __counted_by_or_null(size) local, size_t size);
long long int *__counted_by_or_null(size) use_cbon_non_const_size_non_const_arg_ll(int *buf, size_t size) { // expected-note 6{{pointer 'buf' declared here}}
    // This tests the pointee size types (sizeof(int) < sizeof(long long int)) not matching.

    // expected-note@+3 6{{pointer 'local' declared here}}
    // expected-note@+2 3{{__single parameter 'buf' used to initialize 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but cast of 'local' to 'long long *__bidi_indexable' has pointee type 'long long' (8 bytes)}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'long long *__single __counted_by_or_null(size)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 0 will trap}}
    cbon_non_const_size_ll((long long int*) local, size);

    size_t size2 = size;
    // expected-warning@+2{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'long long *__single __counted_by_or_null(size2)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 0 will trap}}
    long long int *__counted_by_or_null(size2) local2 = (long long int*) local;

    // expected-warning@+2{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'long long *__single __counted_by_or_null(size)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 0 will trap}}
    // expected-warning@+1{{explicit cast of __bidi_indexable 'local' to a larger pointee type creates an out-of-bounds pointer. Later uses of the result may trap}}
    return (long long int*) local;
}

void cbon_non_const_size_c(char* __counted_by_or_null(size) local, size_t size);
char *__counted_by_or_null(size) use_cbon_non_const_size_non_const_arg_c(int *buf, size_t size) { // expected-note 3{{pointer 'buf' declared here}}
    // This tests the pointee size types (sizeof(int) > sizeof(char)) not matching.

    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'char *__single __counted_by_or_null(size)' (aka 'char *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 4 will trap}}
    cbon_non_const_size_c((char*) local, size);

    size_t size2 = size;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'char *__single __counted_by_or_null(size2)' (aka 'char *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 4 will trap}}
    char *__counted_by_or_null(size2) local2 = (char*) local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'char *__single __counted_by_or_null(size)' (aka 'char *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 4 will trap}}
    return (char*) local;
}

//==============================================================================
// Non const size in type
// __sized_by/__sized_by_or_null
//==============================================================================

void sb_non_const_size(void *__sized_by(size) buf, size_t size) {}
void sbon_non_const_size(void *__sized_by_or_null(size) buf, size_t size) {}

// Check the size because the warning text mentions the size explicitly.
_Static_assert(sizeof(int) == 4, "int has unexpected size");

// - if ptr != null then traps if size > sizeof(int) and size != 0
// - if ptr null then traps if size != 0
void *__sized_by(size) use_sb_non_const_size_non_const_arg(int *buf, size_t size) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by(size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    sb_non_const_size(local, size);

    size_t size2 = size;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    void *__sized_by(size2) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by(size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    return local;
}



// - if ptr != null then traps if size > sizeof(int) and size != 0
// - if ptr null then won't trap
void *__sized_by_or_null(size) use_sbon_non_const_size_non_const_arg(int *buf, size_t size) { // expected-note 3{{pointer 'buf' declared here}}
    // expected-note@+2 3{{pointer 'local' declared here}}
    // expected-note@+1 3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by_or_null(size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    sbon_non_const_size(local, size);

    size_t size2 = size;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by_or_null(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    void *__sized_by_or_null(size2) local2 = local;

    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by_or_null(size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    return local;
}


// not a trap
void *__sized_by(*size) use_sb_non_const_size_zero_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // not a trap
    sb_non_const_size(local, 0);

    // False positive
    // The analysis doesn't know that `size` is 0
    size_t size2 = 0;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    void *__sized_by(size2) local2 = local;

    // The analysis knows that `size3` is 0 so it doesn't warn in this case.
    const size_t size3 = 0;
    void *__sized_by(size3) local3 = local;

    // False positive
    // The analysis doesn't know that `*size` is 0
    *size = 0;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by(*size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    return local;
}


// not a trap
void* __sized_by_or_null(*size) use_sbon_non_const_size_zero_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // not a trap
    sbon_non_const_size(local, 0);

    // False positive
    // The analysis doesn't know that `size` is 0
    size_t size2 = 0;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by_or_null(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    void* __sized_by_or_null(size2) local2 = local;

    // The analysis knows that `size3` is 0 so it doesn't warn in this case.
    const size_t size3 = 0;
    void* __sized_by_or_null(size3) local3 = local;

    // False positive
    // The analysis doesn't know that `*size` is 0
    *size = 0;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by_or_null(*size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    return local;
}



// traps only if `local` is null
void* __sized_by(*size) use_sb_non_const_sizeof_int_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // traps only if `local` is null. We currently don't warn about this.
    sb_non_const_size(local, sizeof(int));

    // False positive
    // The analysis doesn't know that `size` is `sizeof(int)`
    size_t size2 = sizeof(int);
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    void* __sized_by(size2) local2 = local;

    // traps only if `local` is null. We currently don't warn about this.
    // The analysis knows that `size3` is `sizeof(int)` so it doesn't warn in
    // this case.
    const size_t size3 = sizeof(int);
    void* __sized_by(size3) local3 = local;


    // False positive
    // The analysis doesn't know that `*size` is `sizeof(int)`
    *size = sizeof(int);
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by(*size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    return local;
}


// not a trap
void* __sized_by_or_null(*size) use_sbon_non_const_sizeof_int_arg(int *buf, size_t *size) { // expected-note 2{{pointer 'buf' declared here}}
    // expected-note@+2 2{{pointer 'local' declared here}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;

    // not a trap
    sbon_non_const_size(local, sizeof(int));

    // False positive
    // The analysis doesn't know that `size` is `sizeof(int)`
    size_t size2 = sizeof(int);
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by_or_null(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    void* __sized_by_or_null(size2) local2 = local;

    const size_t size3 = sizeof(int);
    // The analysis knows that `size3` is `sizeof(int)` so it doesn't warn in
    // this case.
    void* __sized_by_or_null(size3) local3 = local;

    // False positive
    // The analysis doesn't know that `*size` is `sizeof(int)`
    *size = sizeof(int);
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by_or_null(*size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    return local;
}



// trap
void* __sized_by(*size) use_sb_non_const_sizeof_int_plus_one_arg(int *buf, size_t *size) { // expected-note 4{{pointer 'buf' declared here}}
    // expected-note@+4 4{{pointer 'local' declared here}}
    // expected-note@+3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'void *__single __sized_by(size)' (aka 'void *__single') requires 5 bytes or more}}
    // expected-note@+2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'void *__single __sized_by(5UL)' (aka 'void *__single') requires 5 bytes or more}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap when converting to 'void *__single __sized_by(size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    sb_non_const_size(local, sizeof(int)+1);

    // The analysis isn't aware that `size` is `sizeof(int)+1` so the warning
    // not as specific as it should be.
    size_t size2 = sizeof(int)+1;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    void* __sized_by(size2) local2 = local;

    // The analysis is aware that `size` is `sizeof(int)+1` so the warning is
    // more specific.
    const size_t size3 = sizeof(int)+1;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap when converting to 'void *__single __sized_by(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    void* __sized_by(size3) local3 = local;

    // The analysis isn't aware that `*size` is `sizeof(int)+1` so the warning
    // not as specific as it should be.
    *size = sizeof(int)+1;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by(*size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap. If 'local' is null then any size != 0 will trap}}
    return local;
}


// traps iff ptr is not null
void* __sized_by_or_null(*size) use_sbon_non_const_sizeof_int_plus_one_arg(int *buf, size_t *size) { // expected-note 4{{pointer 'buf' declared here}}
    // expected-note@+4 4{{pointer 'local' declared here}}
    // expected-note@+3{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'void *__single __sized_by_or_null(size)' (aka 'void *__single') requires 5 bytes or more}}
    // expected-note@+2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but conversion of 'local' to 'void *__single __sized_by_or_null(5UL)' (aka 'void *__single') requires 5 bytes or more}}
    // expected-note@+1 2{{__single parameter 'buf' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    int *local = buf;
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'void *__single __sized_by_or_null(size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    sbon_non_const_size(local, sizeof(int)+1);

    // The analysis isn't aware that `size` is `sizeof(int)+1` so the warning
    // not as specific as it should be.
    size_t size2 = sizeof(int)+1;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will likely trap when converting to 'void *__single __sized_by_or_null(size2)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    void* __sized_by_or_null(size2) local2 = local;

    // The analysis is aware that `size` is `sizeof(int)+1` so the warning is
    // more specific.
    const size_t size3 = sizeof(int)+1;
    // expected-warning@+1{{assigning from __bidi_indexable local variable 'local' will trap (unless 'local' is null) when converting to 'void *__single __sized_by_or_null(5UL)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer}}
    void* __sized_by_or_null(size3) local3 = local;

    // The analysis isn't aware that `*size` is `sizeof(int)+1` so the warning
    // not as specific as it should be.
    *size = sizeof(int)+1;
    // expected-warning@+1{{returning __bidi_indexable local variable 'local' will likely trap in a future compiler version when converting to 'void *__single __sized_by_or_null(*size)' (aka 'void *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any size > 4 will trap}}
    return local;
}

//==============================================================================
// Multiple __single assignments
//
// This is non-exhaustive due to the huge number of test cases we'd need to
// write.
//==============================================================================

int* single_global; // expected-note{{pointer 'single_global' declared here}}
struct StructWithSinglePtr {
    int* ptr; // expected-note{{StructWithSinglePtr::ptr declared here}}
    int* arr[4]; // expected-note{{StructWithSinglePtr::arr declared here}}
};
int* ret_single(); // expected-note{{'ret_single' declared here}}

void cb_test_multiple_single_assignees_same_size(
    int* p, // expected-note{{pointer 'p' declared here}}
    struct StructWithSinglePtr s,
    size_t size) {
    int* single_ptrs[4] = {0}; // expected-note{{single_ptrs' declared here}}
    int* local; // expected-note{{pointer 'local' declared here}}

    // expected-note@+1{{__single parameter 'p' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    local = p;
    // expected-note@+1{{__single struct member 'ptr' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    local = s.ptr;
    // expected-note@+1{{__single global 'single_global' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    local = single_global;
    // expected-note@+1{{__single element from array 'single_ptrs' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    local = single_ptrs[0];
    // expected-note@+1{{__single return value from call to 'ret_single' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
    local = ret_single();
    // FIXME: This text is slightly wrong. `arr` isn't being assigned directly.
    // expected-note@+1{{__single struct member 'arr' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes}}
    local = s.arr[0];

    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'int *__single __counted_by(size)' (aka 'int *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 1 will trap. If 'local' is null then any count != 0 will trap}}
    cb_non_const_size(local, size);
}

_Static_assert(sizeof(long long int) > sizeof(int), "unexpected size diff");
void consume_one_ll_int(long long int* __counted_by(1));
void cb_test_multiple_single_assignees_mixed_sizes_suppresses_warning(int* p, long long int* q, int cond) {
    long long int* local;
    if (cond)
        local = q;
    else
        local = (long long int* __bidi_indexable)(int* __bidi_indexable) p; // out-of-bounds pointer

    // False negative: The analysis conservatively assumes that `local` takes the
    // largest bounds. In this case the assignment of `q` has the largest bounds
    // which would not lead to a trap. The assignment of `p` would be a trap
    consume_one_ll_int(local);
}

void consume_dynamic_ll_int(long long int* __counted_by(size), size_t size);
void cb_test_multiple_single_assignees_mixed_sizes_warns(
    char* p, // expected-note 2{{pointer 'p' declared here}}
    int* q, // expected-note 2{{pointer 'q' declared here}}
    int cond, size_t size) {
    long long int* local; // expected-note 2{{pointer 'local' declared here}}
    if (cond) {
        // expected-note@+2{{__single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes) but 'local' has pointee type 'long long' (8 bytes)}}
        // expected-note@+1{{_single parameter 'q' assigned to 'local' here results in 'local' having the bounds of a single 'int' (4 bytes)}}
        local = (long long int* __bidi_indexable)(int* __bidi_indexable) q; // out-of-bounds pointer
    }
    else {
        // expected-note@+2{{__single parameter 'p' assigned to 'local' here results in 'local' having the bounds of a single 'char' (1 bytes) but 'local' has pointee type 'long long' (8 bytes)}}
        // expected-note@+1{{_single parameter 'p' assigned to 'local' here results in 'local' having the bounds of a single 'char' (1 bytes)}}
        local = (long long int* __bidi_indexable)(char* __bidi_indexable) p; // out-of-bounds pointer
    }

    // expected-warning@+2{{passing __bidi_indexable 'local' will pass an out-of-bounds pointer. At runtime 'local' is assigned a __single pointer that results in 'local' having bounds smaller than a single 'long long'}}
    // expected-warning@+1{{passing __bidi_indexable local variable 'local' will likely trap when converting to 'long long *__single __counted_by(size)' (aka 'long long *__single') due to 'local' having the bounds of a __single pointer. If 'local' is non-null then any count > 0 will trap. If 'local' is null then any count != 0 will trap}}
    consume_dynamic_ll_int(local, size);
}

//==============================================================================
// __unsafe_forge_bidi_indexable
//==============================================================================

void consume_sized_by_void(void* __sized_by(size), size_t size);
void void_bidi(void * p) {
    // False negative: This will trap. The analysis currently doesn't
    // understand pointers that come from `__unsafe_forge_bidi_indexable`.
    //
    // This isn't really `__single` so it's not surprising the analysis does
    // not understand this.
    void* local = __unsafe_forge_bidi_indexable(void*, p, sizeof(char));
    consume_sized_by_void(local, sizeof(int));
}


