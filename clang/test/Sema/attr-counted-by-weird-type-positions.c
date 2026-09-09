// RUN: %clang_cc1 -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

// ============================================================================
// SIMPLE POINTER: int *buf
// ============================================================================

// Position: after *, before identifier
// Applies to `int *`.
struct ptr_after_star {
  int count;
  int *__counted_by(count) buf;
};

// Position: before type specifier
// Applies to the top-level type.
struct ptr_before_type {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) int *buf;
};

// Position: after type, before *
// Applies to `int`.
struct ptr_after_type {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) *buf;
};

// Position: after identifier
// Applies to the top-level type.
struct ptr_after_ident {
  int count;
  int *buf __counted_by(count);
};

// ============================================================================
// TYPEDEF POINTER: ptr_to_int_t buf
// ============================================================================

typedef int * ptr_to_int_t;

// Position: after typedef name, before identifier
// Applies to `ptr_to_int_t`.
struct typedef_after_type {
  int count;
  ptr_to_int_t __counted_by(count) buf;
};

// Position: before typedef name
// Applies to the top-level type.
struct typedef_before_type {
  int count;
  __counted_by(count) ptr_to_int_t buf;
};

// Position: after identifier
// Applies to the top-level type.
struct typedef_after_ident {
  int count;
  ptr_to_int_t buf __counted_by(count);
};

// ============================================================================
// POINTER TO ARRAY: int (*buf)[4]
// ============================================================================

// Position: after type, before (*...)
// Applies to `int`.
struct ptr_to_arr_after_type {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* buf)[4];
};

// Position: before type
// Applies to the top-level type.
struct ptr_to_arr_before_type {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) int (* buf)[4];
};

// Position: after *, before identifier (inside parens)
// Applies to `int (*)[4]`.
struct ptr_to_arr_after_star {
  int count;
  int (* __counted_by(count) buf)[4];
};

// Position: after identifier, before ) (inside parens)
// Invalid position - causes parse error
struct ptr_to_arr_after_ident {
  int count;
  int (*buf __counted_by(count))[4]; // Invalid position
  // expected-error@-1{{expected ')'}}
  // expected-note@-2{{to match this '('}}
};

// Position: after [4]
// Applies to the top-level type.
struct ptr_to_arr_after_brackets {
  int count;
  int (* buf)[4] __counted_by(count);
};

// Position: after (, before *
struct ptr_to_arr_after_lparen {
  int count;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int (__counted_by(count) *buf)[4];
};

// Position: inside [4]
struct ptr_to_arr_inside_brackets {
  int count;
  int (* buf)[4 __counted_by(count)]; // Invalid syntax
  // expected-error@-1{{expected ']'}}
  // expected-note@-2{{to match this '['}}
};

// Position: before [4]
struct ptr_to_arr_before_brackets {
  int count;
  // expected-error@+1{{expected ';' at end of declaration list}}
  int (* buf) __counted_by(count) [4]; // Invalid syntax
};

// Position: double parens, after ((, before *
struct ptr_to_arr_double_paren1 {
  int count;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int ((__counted_by(count) * buf))[4];
};

// Position: double parens, after *, before identifier
struct ptr_to_arr_double_paren2 {
  int count;
  int ((* __counted_by(count) buf))[4];
};

// ============================================================================
// POINTER TO ARRAY WITH QUALIFIERS
// ============================================================================

// const pointer
struct ptr_to_arr_const_ptr1 {
  int count;
  int (* const __counted_by(count) buf)[4];
};

struct ptr_to_arr_const_ptr2 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* const buf)[4];
};

// pointer to const
struct ptr_to_arr_ptr_to_const {
  int count;
  const int (* __counted_by(count) buf)[4];
};

struct ptr_to_arr_ptr_to_const2 {
  int count;
  int const (* __counted_by(count) buf)[4];
};

// restrict pointer
struct ptr_to_arr_restrict1 {
  int count;
  int (* __restrict __counted_by(count) buf)[4];
};

struct ptr_to_arr_restrict2 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* __restrict buf)[4];
};

// ============================================================================
// POINTER TO MULTI-DIMENSIONAL ARRAY: int (*buf)[4][8]
// ============================================================================

struct ptr_to_multidim_arr_after_type {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* buf)[4][8];
};

struct ptr_to_multidim_arr_after_star {
  int count;
  int (* __counted_by(count) buf)[4][8];
};

struct ptr_to_multidim_arr_middle {
  int count;
  // expected-error@+1{{expected ';' at end of declaration list}}
  int (* buf)[4] __counted_by(count) [8]; // Invalid position
};

struct ptr_to_multidim_arr_after_all {
  int count;
  int (* buf)[4][8] __counted_by(count);
  // This doesn't trigger an error - the attribute applies to the pointer
};

// ============================================================================
// ARRAY OF POINTERS TO ARRAY: int (*buf[10])[4]
// ============================================================================

struct arr_of_ptr_to_arr_after_type {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* buf[10])[4];
};

struct arr_of_ptr_to_arr_after_star {
  int count;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int (* __counted_by(count) buf[10])[4];
};

struct arr_of_ptr_to_arr_middle {
  int count;
  // expected-error@+2{{'counted_by' on arrays only applies to C99 flexible array members}}
  // expected-error@+1{{expected ';' at end of declaration list}}
  int (* buf[10]) __counted_by(count) [4]; // Invalid position
};

struct arr_of_ptr_to_arr_inside_first_brackets {
  int count;
  int (* buf __counted_by(count) [10])[4];
  // expected-error@-1{{expected ')'}}
  // expected-note@-2{{to match this '('}}
};

// ============================================================================
// TYPEDEF ARRAY: arr4_t *buf where arr4_t is int[4]
// ============================================================================

typedef int arr4_t[4];

struct typedef_arr_before_type {
  int count;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  __counted_by(count) arr4_t * buf;
};

struct typedef_arr_after_type {
  int count;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  arr4_t __counted_by(count) * buf;
};

struct typedef_arr_after_star {
  int count;
  arr4_t * __counted_by(count) buf;
};

// ============================================================================
// FUNCTION POINTER: int (*buf)(void)
// ============================================================================

// Position: after *, before identifier
struct fptr_after_star {
  int count;
  // expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'int (void)' is a function type}}
  int (* __counted_by(count) buf)(void);
};

// Position: after (, before *
struct fptr_after_lparen {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int (__counted_by(count) *buf)(void);
};

// ============================================================================
// _ATOMIC POINTER VARIATIONS
// ============================================================================

// _Atomic(int *) - atomic pointer type
struct atomic_ptr_type {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  _Atomic(int *) __counted_by(count) buf;
};

// Attribute inside _Atomic (likely invalid)
struct atomic_ptr_attr_inside {
  int count;
  _Atomic(int *__counted_by(count)) buf;
};

struct atomic_ptr_attr_inside_no_forward_ref {
  int count;
  // FIXME: should not be allowed
  _Atomic(int *__counted_by(count)) buf;
};

struct atomic_ptr_attr_after {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) _Atomic(int *) buf;
};

struct atomic_ptr_attr_after_no_forward_ref {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) _Atomic(int *) buf;
};

// _Atomic int * - could be atomic int or atomic pointer
struct atomic_ambiguous {
  int count;
  _Atomic int * __counted_by(count) buf;
};

// int *_Atomic - atomic pointer (unambiguous)
struct atomic_ptr_unambiguous1 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int *_Atomic __counted_by(count) buf;
};

// __counted_by before _Atomic
struct atomic_ptr_attr_before_atomic1 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int *__counted_by(count) _Atomic buf;
};

// __counted_by before * _Atomic
struct atomic_ptr_attr_before_atomic2 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) * _Atomic buf;
};

// _Atomic before type
struct atomic_ptr_atomic_first1 {
  int count;
  _Atomic int *__counted_by(count) buf;
};

// _Atomic before type, attribute after *
struct atomic_ptr_atomic_first2 {
  int count;
  _Atomic int * __counted_by(count) buf;
};

// __counted_by at the end
struct atomic_ptr_attr_at_end1 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int *_Atomic buf __counted_by(count);
};

// __counted_by at the end with space
struct atomic_ptr_attr_at_end2 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic buf __counted_by(count);
};

// ============================================================================
// _ATOMIC POINTER TO ARRAY
// ============================================================================

struct atomic_ptr_to_arr1 {
  int count;
  _Atomic int (* __counted_by(count) buf)[4];
};

struct atomic_ptr_to_arr2 {
  int count;
  // expected-error@+2{{expected a type}}
  // expected-error@+1{{expected member name or ';' after declaration specifiers}}
  int _Atomic (* __counted_by(count) buf)[4];
};

struct atomic_ptr_to_arr3 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int (* _Atomic __counted_by(count) buf)[4];
};

// ============================================================================
// ATOMIC WITH CONST/VOLATILE/RESTRICT QUALIFIERS
// ============================================================================

// const _Atomic pointer
struct atomic_const_ptr1 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * const _Atomic __counted_by(count) buf;
};

struct atomic_const_ptr2 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic const __counted_by(count) buf;
};

struct atomic_const_ptr3 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  const int * _Atomic __counted_by(count) buf;
};

// volatile _Atomic pointer
struct atomic_volatile_ptr1 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * volatile _Atomic __counted_by(count) buf;
};

struct atomic_volatile_ptr2 {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic volatile __counted_by(count) buf;
};

// restrict _Atomic pointer
struct atomic_restrict_ptr1 {
  int count;
  // expected-error@+2{{restrict requires a pointer or reference ('_Atomic(int *)' is invalid)}}
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * __restrict _Atomic __counted_by(count) buf;
};

struct atomic_restrict_ptr2 {
  int count;
  // expected-error@+2{{restrict requires a pointer or reference ('_Atomic(int *)' is invalid)}}
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic __restrict __counted_by(count) buf;
};

// Combined qualifiers
struct atomic_const_volatile_ptr {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * const volatile _Atomic __counted_by(count) buf;
};

struct atomic_all_qualifiers {
  int count;
  // expected-error@+2{{restrict requires a pointer or reference ('_Atomic(int *)' is invalid)}}
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * const volatile __restrict _Atomic __counted_by(count) buf;
};
