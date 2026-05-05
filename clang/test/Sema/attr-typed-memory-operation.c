// RUN: %clang_cc1 -ftyped-memory-operations -triple arm64-apple-ios -Rtmo-remarks -fsyntax-only -verify %s
// RUN: %clang_cc1 -ftyped-memory-operations -triple arm64-apple-ios -fsyntax-only -verify %s

#define _TYPED_ALLOC(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_alloc1(__SIZE_TYPE__ size, unsigned long long descriptor);
// expected-note@-1 {{rewrite target}}
// expected-note@-2 {{rewrite target}}
void typed_alloc2(__SIZE_TYPE__ size, unsigned long long descriptor, void **out);
// expected-note@-1 {{rewrite target}}
void typed_alloc3(void** out, __SIZE_TYPE__ size, unsigned long long descriptor);
void *incorrect_descriptor(__SIZE_TYPE__ size, char descriptor);
// expected-note@-1 {{rewrite target here}}

void *alloc1(__SIZE_TYPE__ size) _TYPED_ALLOC(typed_alloc1, 1);
void *alloc2(__SIZE_TYPE__ size) _TYPED_ALLOC(incorrect_descriptor, 1);
// expected-error@-1 {{typed memory operation 'incorrect_descriptor' has incompatible type ('void *(unsigned long, unsigned long)' vs 'void *(unsigned long, char)')}}
void *alloc3(__SIZE_TYPE__ size) _TYPED_ALLOC(missing, 1);
// expected-error@-1 {{use of undeclared identifier 'missing'}}
void *alloc4(__SIZE_TYPE__ size) _TYPED_ALLOC(typed_alloc2, 1);
// expected-error@-1 {{typed memory operation 'typed_alloc2' has incompatible type ('void *(unsigned long, unsigned long)' vs 'void (unsigned long, unsigned long long, void **)')}}
void alloc5(__SIZE_TYPE__ size, void **out) _TYPED_ALLOC(typed_alloc2, 1);
void alloc6(__SIZE_TYPE__ size, void **out) _TYPED_ALLOC(typed_alloc2, 2);
// expected-error@-1 {{invalid parameter type for inference at index 2. 'void **' is not an integer type}}
void alloc7(__SIZE_TYPE__ size, void **out) _TYPED_ALLOC(typed_alloc1, 1);
// expected-error@-1 {{typed memory operation 'typed_alloc1' has incompatible type ('void (unsigned long, unsigned long, void **)' vs 'void *(unsigned long, unsigned long long)')}}
void alloc8(void **out, __SIZE_TYPE__ size) _TYPED_ALLOC(typed_alloc3, 2);
void alloc9(void **out, __SIZE_TYPE__ size) _TYPED_ALLOC(typed_alloc1, 2);
// expected-error@-1 {{typed memory operation 'typed_alloc1' has incompatible type ('void (void **, unsigned long, unsigned long)' vs 'void *(unsigned long, unsigned long long)')}}

int wrong_thing;
void alloc10(__SIZE_TYPE__) _TYPED_ALLOC(wrong_thing, 1);
// expected-error@-1 {{typed memory operation 'wrong_thing' must be a function}}

void alloc11(__SIZE_TYPE__) _TYPED_ALLOC(1, 1);
// expected-error@-1 {{typed memory operation '1' must be a function}}

typedef int __attribute__((vector_size(16))) ivector16;
typedef int __attribute__((vector_size(16))) ivector16_2;
typedef int __attribute__((vector_size(32))) ivector32;

ivector16 *typed_ivalloc(__SIZE_TYPE__, unsigned long long descriptor);
// expected-note@-1 2 {{rewrite target here}}
ivector16 *ivalloc1(__SIZE_TYPE__) _TYPED_ALLOC(typed_ivalloc, 1);
ivector16_2 *ivalloc2(__SIZE_TYPE__) _TYPED_ALLOC(typed_ivalloc, 1);
ivector32 *ivalloc3(__SIZE_TYPE__) _TYPED_ALLOC(typed_ivalloc, 1);
// expected-error@-1 {{typed memory operation 'typed_ivalloc' has incompatible type ('ivector32 *(unsigned long, unsigned long)' vs 'ivector16 *(unsigned long, unsigned long long)')}}
int *ivalloc4(__SIZE_TYPE__) _TYPED_ALLOC(typed_ivalloc, 1);
// expected-error@-1 {{typed memory operation 'typed_ivalloc' has incompatible type ('int *(unsigned long, unsigned long)' vs 'ivector16 *(unsigned long, unsigned long long)')}}

// clang doesn't see pointer alignment as part of the type, so these
// types are considered the same. It's possible we'll want to check
// for this in future given we are after all an allocation related
// attribute.
typedef int *__attribute__((aligned(16))) aligned16_ptr;
typedef int *__attribute__((aligned(32))) aligned32_ptr;

aligned16_ptr typed_aligned_alloc(__SIZE_TYPE__, unsigned long long descriptor);
aligned16_ptr aligned_alloc16(__SIZE_TYPE__) _TYPED_ALLOC(typed_aligned_alloc, 1);
aligned32_ptr aligned_alloc32(__SIZE_TYPE__) _TYPED_ALLOC(typed_aligned_alloc, 1);

typedef int _Atomic * atomic_ptr;
atomic_ptr typed_atomic_alloc(__SIZE_TYPE__, unsigned long long descriptor);
// expected-note@-1 {{rewrite target here}}
atomic_ptr atomic_alloc(__SIZE_TYPE__) _TYPED_ALLOC(typed_atomic_alloc, 1);
int *unatomic_alloc(__SIZE_TYPE__) _TYPED_ALLOC(typed_atomic_alloc, 1);
// expected-error@-1 {{typed memory operation 'typed_atomic_alloc' has incompatible type ('int *(unsigned long, unsigned long)' vs 'atomic_ptr (unsigned long, unsigned long long)' (aka '_Atomic(int) *(unsigned long, unsigned long long)'))}}

