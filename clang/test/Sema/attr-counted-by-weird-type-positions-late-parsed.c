// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

// ============================================================================
// SIMPLE POINTER: int *buf
// ============================================================================

// Position: after *, before identifier
// Applies to `int *`.
struct ptr_after_star {
  int *__counted_by(count) buf;
  int count;
};

// Position: before type specifier
// Applies to the top-level type.
struct ptr_before_type {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) int *buf;
  int count;
};

// Position: after type, before *
// Applies to `int`.
struct ptr_after_type {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) *buf;
  int count;
};

// Position: after identifier
// Applies to the top-level type.
struct ptr_after_ident {
  int *buf __counted_by(count);
  int count;
};

// ============================================================================
// TYPEDEF POINTER: ptr_to_int_t buf
// ============================================================================

typedef int * ptr_to_int_t;

// Position: after typedef name, before identifier
// Applies to `ptr_to_int_t`.
struct typedef_after_type {
  ptr_to_int_t __counted_by(count) buf;
  int count;
};

// Position: before typedef name
// Applies to the top-level type.
struct typedef_before_type {
  __counted_by(count) ptr_to_int_t buf;
  int count;
};

// Position: after identifier
// Applies to the top-level type.
struct typedef_after_ident {
  ptr_to_int_t buf __counted_by(count);
  int count;
};

// ============================================================================
// POINTER TO ARRAY: int (*buf)[4]
// ============================================================================

// Position: after type, before (*...)
// Applies to `int`.
struct ptr_to_arr_after_type {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* buf)[4];
  int count;
};

// Position: before type
// Applies to the top-level type.
struct ptr_to_arr_before_type {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) int (* buf)[4];
  int count;
};

// Position: after *, before identifier (inside parens)
// Applies to `int (*)[4]`.
struct ptr_to_arr_after_star {
  int (* __counted_by(count) buf)[4];
  int count;
};

// Position: after identifier, before ) (inside parens)
// Invalid position - causes parse error
struct ptr_to_arr_after_ident {
  int (*buf __counted_by(count))[4]; // Invalid position
  // expected-error@-1{{expected ')'}}
  // expected-note@-2{{to match this '('}}
  int count;
};

// Position: after [4]
// Applies to the top-level type.
struct ptr_to_arr_after_brackets {
  int (* buf)[4] __counted_by(count);
  int count;
};

// Position: after (, before *
struct ptr_to_arr_after_lparen {
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int (__counted_by(count) *buf)[4];
  int count;
};

// Position: inside [4]
struct ptr_to_arr_inside_brackets {
  int (* buf)[4 __counted_by(count)]; // Invalid syntax
  // expected-error@-1{{expected ']'}}
  // expected-note@-2{{to match this '['}}
  int count;
};

// Position: before [4]
struct ptr_to_arr_before_brackets {
  // expected-error@+1{{expected ';' at end of declaration list}}
  int (* buf) __counted_by(count) [4]; // Invalid syntax
  int count;
};

// Position: double parens, after ((, before *
struct ptr_to_arr_double_paren1 {
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int ((__counted_by(count) * buf))[4];
  int count;
};

// Position: double parens, after *, before identifier
struct ptr_to_arr_double_paren2 {
  int ((* __counted_by(count) buf))[4];
  int count;
};

// ============================================================================
// POINTER TO ARRAY WITH QUALIFIERS
// ============================================================================

// const pointer
struct ptr_to_arr_const_ptr1 {
  int (* const __counted_by(count) buf)[4];
  int count;
};

struct ptr_to_arr_const_ptr2 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* const buf)[4];
  int count;
};

// pointer to const
struct ptr_to_arr_ptr_to_const {
  const int (* __counted_by(count) buf)[4];
  int count;
};

struct ptr_to_arr_ptr_to_const2 {
  int const (* __counted_by(count) buf)[4];
  int count;
};

// restrict pointer
struct ptr_to_arr_restrict1 {
  int (* __restrict __counted_by(count) buf)[4];
  int count;
};

struct ptr_to_arr_restrict2 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* __restrict buf)[4];
  int count;
};

// ============================================================================
// POINTER TO MULTI-DIMENSIONAL ARRAY: int (*buf)[4][8]
// ============================================================================

struct ptr_to_multidim_arr_after_type {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* buf)[4][8];
  int count;
};

struct ptr_to_multidim_arr_after_star {
  int (* __counted_by(count) buf)[4][8];
  int count;
};

struct ptr_to_multidim_arr_middle {
  // expected-error@+1{{expected ';' at end of declaration list}}
  int (* buf)[4] __counted_by(count) [8]; // Invalid position
  int count;
};

struct ptr_to_multidim_arr_after_all {
  int (* buf)[4][8] __counted_by(count);
  // This doesn't trigger an error - the attribute applies to the pointer
  int count;
};

// ============================================================================
// ARRAY OF POINTERS TO ARRAY: int (*buf[10])[4]
// ============================================================================

struct arr_of_ptr_to_arr_after_type {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) (* buf[10])[4];
  int count;
};

struct arr_of_ptr_to_arr_after_star {
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int (* __counted_by(count) buf[10])[4];
  int count;
};

struct arr_of_ptr_to_arr_middle {
  // expected-error@+2{{'counted_by' on arrays only applies to C99 flexible array members}}
  // expected-error@+1{{expected ';' at end of declaration list}}
  int (* buf[10]) __counted_by(count) [4]; // Invalid position
  int count;
};

struct arr_of_ptr_to_arr_inside_first_brackets {
  int (* buf __counted_by(count) [10])[4];
  // expected-error@-1{{expected ')'}}
  // expected-note@-2{{to match this '('}}
  int count;
};

// ============================================================================
// TYPEDEF ARRAY: arr4_t *buf where arr4_t is int[4]
// ============================================================================

typedef int arr4_t[4];

struct typedef_arr_before_type {
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  __counted_by(count) arr4_t * buf;
  int count;
};

struct typedef_arr_after_type {
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  arr4_t __counted_by(count) * buf;
  int count;
};

struct typedef_arr_after_star {
  arr4_t * __counted_by(count) buf;
  int count;
};

// ============================================================================
// FUNCTION POINTER: int (*buf)(void)
// ============================================================================

// Position: after *, before identifier
struct fptr_after_star {
  // expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'int (void)' is a function type}}
  int (* __counted_by(count) buf)(void);
  int count;
};

// Position: after (, before *
struct fptr_after_lparen {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int (__counted_by(count) *buf)(void);
  int count;
};

// ============================================================================
// _ATOMIC POINTER VARIATIONS
// ============================================================================

// _Atomic(int *) - atomic pointer type
struct atomic_ptr_type {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  _Atomic(int *) __counted_by(count) buf;
  int count;
};

// Attribute inside _Atomic (likely invalid)
struct atomic_ptr_attr_inside {
  // expected-error@+1{{use of undeclared identifier 'count'}}
  _Atomic(int *__counted_by(count)) buf;
  int count;
};

struct atomic_ptr_attr_inside_no_forward_ref {
  int count;
  // FIXME: should not be allowed
  _Atomic(int *__counted_by(count)) buf;
};

struct atomic_ptr_attr_after {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) _Atomic(int *) buf;
  int count;
};

struct atomic_ptr_attr_after_no_forward_ref {
  int count;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  __counted_by(count) _Atomic(int *) buf;
};

// _Atomic int * - could be atomic int or atomic pointer
struct atomic_ambiguous {
  _Atomic int * __counted_by(count) buf;
  int count;
};

// int *_Atomic - atomic pointer (unambiguous)
struct atomic_ptr_unambiguous1 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int *_Atomic __counted_by(count) buf;
  int count;
};

// __counted_by before _Atomic
struct atomic_ptr_attr_before_atomic1 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int *__counted_by(count) _Atomic buf;
  int count;
};

// __counted_by before * _Atomic
struct atomic_ptr_attr_before_atomic2 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(count) * _Atomic buf;
  int count;
};

// _Atomic before type
struct atomic_ptr_atomic_first1 {
  _Atomic int *__counted_by(count) buf;
  int count;
};

// _Atomic before type, attribute after *
struct atomic_ptr_atomic_first2 {
  _Atomic int * __counted_by(count) buf;
  int count;
};

// __counted_by at the end
struct atomic_ptr_attr_at_end1 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int *_Atomic buf __counted_by(count);
  int count;
};

// __counted_by at the end with space
struct atomic_ptr_attr_at_end2 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic buf __counted_by(count);
  int count;
};

// ============================================================================
// _ATOMIC POINTER TO ARRAY
// ============================================================================

struct atomic_ptr_to_arr1 {
  _Atomic int (* __counted_by(count) buf)[4];
  int count;
};

struct atomic_ptr_to_arr2 {
  // expected-error@+3{{expected a type}}
  // expected-error@+2{{use of undeclared identifier 'count'}}
  // expected-error@+1{{expected member name or ';' after declaration specifiers}}
  int _Atomic (* __counted_by(count) buf)[4];
  int count;
};

struct atomic_ptr_to_arr3 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int (* _Atomic __counted_by(count) buf)[4];
  int count;
};

// ============================================================================
// ATOMIC WITH CONST/VOLATILE/RESTRICT QUALIFIERS
// ============================================================================

// const _Atomic pointer
struct atomic_const_ptr1 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * const _Atomic __counted_by(count) buf;
  int count;
};

struct atomic_const_ptr2 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic const __counted_by(count) buf;
  int count;
};

struct atomic_const_ptr3 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  const int * _Atomic __counted_by(count) buf;
  int count;
};

// volatile _Atomic pointer
struct atomic_volatile_ptr1 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * volatile _Atomic __counted_by(count) buf;
  int count;
};

struct atomic_volatile_ptr2 {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic volatile __counted_by(count) buf;
  int count;
};

// restrict _Atomic pointer
struct atomic_restrict_ptr1 {
  // expected-error@+2{{restrict requires a pointer or reference ('_Atomic(int *)' is invalid)}}
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * __restrict _Atomic __counted_by(count) buf;
  int count;
};

struct atomic_restrict_ptr2 {
  // expected-error@+2{{restrict requires a pointer or reference ('_Atomic(int *)' is invalid)}}
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * _Atomic __restrict __counted_by(count) buf;
  int count;
};

// Combined qualifiers
struct atomic_const_volatile_ptr {
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * const volatile _Atomic __counted_by(count) buf;
  int count;
};

struct atomic_all_qualifiers {
  // expected-error@+2{{restrict requires a pointer or reference ('_Atomic(int *)' is invalid)}}
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int * const volatile __restrict _Atomic __counted_by(count) buf;
  int count;
};
