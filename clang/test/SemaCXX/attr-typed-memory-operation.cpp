// RUN: %clang_cc1 -Rtmo-remarks -ftyped-memory-operations -triple arm64-apple-ios -fsyntax-only -verify %s
// RUN: %clang_cc1 -ftyped-memory-operations -triple arm64-apple-ios -fsyntax-only -verify %s

#define _TYPED_ALLOC(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_alloc1(__SIZE_TYPE__ size, unsigned long long descriptor);
// expected-note@-1 3 {{rewrite target here}}
void typed_alloc2(__SIZE_TYPE__ size, unsigned long long descriptor, void **out);
// expected-note@-1 {{rewrite target here}}
void typed_alloc3(void** out, __SIZE_TYPE__ size, unsigned long long descriptor);
// expected-note@-1 {{rewrite target here}}
void *incorrect_descriptor(__SIZE_TYPE__ size, char descriptor);
// expected-note@-1 {{rewrite target here}}
void *typed_alloc_to_shadow(__SIZE_TYPE__ size, unsigned long long);
// expected-note@-1 {{candidate function}}
void *typed_alloc_to_shadow(__SIZE_TYPE__ size, float descriptor);
// expected-note@-1 {{candidate function}}
template <typename T> T* templated_typed_alloc1(__SIZE_TYPE__ size, unsigned long long);
// expected-note@-1 {{rewrite target here}}
template <typename T> void* templated_typed_alloc2(__SIZE_TYPE__ size, T);
// expected-note@-1 {{rewrite target here}}
template <typename T> void* templated_typed_alloc3(T size, unsigned long long);
// expected-note@-1 3 {{rewrite target here}}

void *alloc1(__SIZE_TYPE__ size) _TYPED_ALLOC(typed_alloc1, 1);
void *alloc2(__SIZE_TYPE__ size) _TYPED_ALLOC(incorrect_descriptor, 1);
// expected-error@-1 {{typed memory operation 'incorrect_descriptor' has incompatible type ('void *(unsigned long, unsigned long)' vs 'void *(unsigned long, char)')}}
void *alloc3(__SIZE_TYPE__ size) _TYPED_ALLOC(missing_func, 1);
// expected-error@-1 {{use of undeclared identifier 'missing_func'}}
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
void *alloc10(__SIZE_TYPE__ size) _TYPED_ALLOC(typed_alloc_to_shadow, 1);
// expected-error@-1 {{typed memory operation 'typed_alloc_to_shadow' has multiple overloads}}
int *alloc11(__SIZE_TYPE__ size) _TYPED_ALLOC(templated_typed_alloc1<int>, 1);
float *alloc12(__SIZE_TYPE__ size) _TYPED_ALLOC(templated_typed_alloc1<int>, 1);
// expected-error@-1 {{typed memory operation 'templated_typed_alloc1' has incompatible type ('float *(unsigned long, unsigned long)' vs 'int *(unsigned long, unsigned long long)')}}
void *alloc13(__SIZE_TYPE__ size) _TYPED_ALLOC(templated_typed_alloc2<unsigned long long>, 1);
void *alloc14(__SIZE_TYPE__ size) _TYPED_ALLOC(templated_typed_alloc2<int>, 1);
// expected-error@-1 {{typed memory operation 'templated_typed_alloc2' has incompatible type ('void *(unsigned long, unsigned long)' vs 'void *(unsigned long, int)')}}
void *alloc15(__SIZE_TYPE__ size) _TYPED_ALLOC(templated_typed_alloc3<unsigned long long>, 1);
// expected-error@-1 {{typed memory operation 'templated_typed_alloc3' has incompatible type ('void *(unsigned long, unsigned long)' vs 'void *(unsigned long long, unsigned long long)')}}
void *alloc16(__SIZE_TYPE__ size) _TYPED_ALLOC(templated_typed_alloc3<unsigned>, 1);
// expected-error@-1 {{typed memory operation 'templated_typed_alloc3' has incompatible type ('void *(unsigned long, unsigned long)' vs 'void *(unsigned int, unsigned long long)')}}
void *alloc17(int size) _TYPED_ALLOC(templated_typed_alloc3<unsigned>, 1);
// expected-error@-1 {{typed memory operation 'templated_typed_alloc3' has incompatible type ('void *(int, unsigned long)' vs 'void *(unsigned int, unsigned long long)')}}
void *alloc18(int size) _TYPED_ALLOC(typed_alloc1, 1);
// expected-error@-1 {{typed memory operation 'typed_alloc1' has incompatible type ('void *(int, unsigned long)' vs 'void *(unsigned long, unsigned long long)')}}
void alloc19(char** out, __SIZE_TYPE__ size) _TYPED_ALLOC(typed_alloc3, 2);
// expected-error@-1 {{typed memory operation 'typed_alloc3' has incompatible type ('void (char **, unsigned long, unsigned long)' vs 'void (void **, unsigned long, unsigned long long)')}}
