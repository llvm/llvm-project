// RUN: %clang_cc1 -ftyped-memory-operations -fsyntax-only -verify %s

#define _TYPED_ALLOC(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *malloc(unsigned long);
void *typed_malloc(__SIZE_TYPE__, unsigned long long);
void *calloc(unsigned long, unsigned long);
void *typed_calloc(unsigned, unsigned, unsigned long long);
// expected-note@-1 {{rewrite target here}}

// Some function defs that don't match the required interface
void *invalid_typed_malloc1();
void *invalid_typed_malloc2(double);
// expected-note@-1 {{rewrite target here}}
int invalid_typed_malloc3(unsigned);
// expected-note@-1 {{rewrite target here}}
// expected-note@-2 {{rewrite target here}}
// expected-note@-3 {{rewrite target here}}
int invalid_typed_malloc4(__SIZE_TYPE__, unsigned long long);
void* invalid_typed_malloc5(__SIZE_TYPE__, int);


void *my_malloc2(__SIZE_TYPE__ size) _TYPED_ALLOC(typed_malloc, 1);
void *my_malloc3(double size) _TYPED_ALLOC(typed_malloc, 1);
// expected-error@-1 {{invalid parameter type for inference at index 1. 'double' is not an integer type}}
void *my_malloc4(unsigned size) _TYPED_ALLOC(typed_malloc, -1);
// expected-error@-1 {{'typed_memory_operation' attribute parameter 1 is out of bounds}}
void *my_malloc4(unsigned size) _TYPED_ALLOC(typed_malloc, 0);
// expected-error@-1 {{'typed_memory_operation' attribute parameter 1 is out of bounds}}
void *my_malloc6(unsigned size) _TYPED_ALLOC(typed_malloc, 2);
// expected-error@-1 {{'typed_memory_operation' attribute parameter 1 is out of bounds}}

void *my_malloc_invalid1(unsigned size) _TYPED_ALLOC(invalid_typed_malloc0, 1);
// expected-error@-1 {{use of undeclared identifier 'invalid_typed_malloc0'}}
void *my_malloc_invalid2(unsigned size) _TYPED_ALLOC(invalid_typed_malloc1, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc1' must have a prototype}}
void *my_malloc_invalid3(unsigned size) _TYPED_ALLOC(invalid_typed_malloc2, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc2' has incompatible type ('void *(unsigned int, unsigned long)' vs 'void *(double)')}}
void *my_malloc_invalid4(unsigned size) _TYPED_ALLOC(invalid_typed_malloc3, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc3' has incompatible type ('void *(unsigned int, unsigned long)' vs 'int (unsigned int)')}}
// intentionally using the wrong function
void *my_malloc_invalid5(unsigned size) _TYPED_ALLOC(invalid_typed_malloc3, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc3' has incompatible type ('void *(unsigned int, unsigned long)' vs 'int (unsigned int)')}}
void *my_malloc_invalid6(unsigned size) _TYPED_ALLOC(invalid_typed_malloc3, 1);
// expected-error@-1 {{typed memory operation 'invalid_typed_malloc3' has incompatible type ('void *(unsigned int, unsigned long)' vs 'int (unsigned int)')}}

void *my_malloc12(__SIZE_TYPE__ size) _TYPED_ALLOC(calloc, 1);
void *my_malloc13(unsigned size) _TYPED_ALLOC(typed_calloc, 1);
// expected-error@-1 {{typed memory operation 'typed_calloc' has incompatible type ('void *(unsigned int, unsigned long)' vs 'void *(unsigned int, unsigned int, unsigned long long)')}}

void *my_calloc1(unsigned count, unsigned size) _TYPED_ALLOC(typed_calloc, 2);

// Redeclarations
void *typed_for_redecl(unsigned long long, unsigned long long, unsigned long long);
void *typed_for_redecl2(unsigned long long, unsigned long long, unsigned long long);
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl, 2);
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl, 2);
// expected-note@-1 2 {{conflicting attribute is here}}
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl, 1);
// expected-error@-1 {{attribute 'typed_memory_operation' is already applied with different arguments}}
void *my_calloc2(unsigned long long count, unsigned long long size) _TYPED_ALLOC(typed_for_redecl2, 2);
// expected-error@-1 {{attribute 'typed_memory_operation' is already applied with different arguments}}
