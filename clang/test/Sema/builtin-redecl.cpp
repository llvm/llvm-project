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

void __va_start(__builtin_va_list*, ...);

          void *__builtin_assume_aligned(const void *, size_t, ...);
#ifdef __cplusplus
constexpr void *__builtin_assume_aligned(const void *, size_t, ...);
          void *__builtin_assume_aligned(const void *, size_t, ...) noexcept;
constexpr void *__builtin_assume_aligned(const void *, size_t, ...) noexcept;
          void *__builtin_assume_aligned(const void *, size_t, ...) throw();
constexpr void *__builtin_assume_aligned(const void *, size_t, ...) throw();

// expected-error@+1 {{constexpr declaration of '__builtin_calloc' follows non-constexpr declaration}}
constexpr void *__builtin_calloc(size_t, size_t);
// expected-note@-1 {{previous declaration is here}}
#endif
