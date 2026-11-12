// RUN: %clang_cc1 %s -fsyntax-only -verify 
// RUN: %clang_cc1 %s -fsyntax-only -verify -x c
// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-compatibility

typedef __typeof__(sizeof(0)) size_t;

// Redeclaring library builtins is OK.
void exit(int);

// expected-error@+2 {{cannot redeclare builtin function '__builtin_va_copy'}}
// expected-note@+1 {{'__builtin_va_copy' is a builtin with type}}
void __builtin_va_copy(double d);

// expected-error@+2 {{cannot redeclare builtin function '__builtin_va_end'}}
// expected-note@+1 {{'__builtin_va_end' is a builtin with type}}
void __builtin_va_end(__builtin_va_list);
// RUN: %clang_cc1 %s -fsyntax-only -verify 
// RUN: %clang_cc1 %s -fsyntax-only -verify -x c

void __va_start(__builtin_va_list*, ...);

#ifdef __cplusplus
void *__builtin_assume_aligned(const void *, size_t, ...) noexcept;
#else
void *__builtin_assume_aligned(const void *, size_t, ...);
#endif
