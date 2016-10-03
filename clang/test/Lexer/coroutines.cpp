// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -fcoroutines-ts -DCORO -fsyntax-only %s

#ifdef CORO
#define CORO_KEYWORD(NAME) _Static_assert(!__is_identifier(NAME), #NAME)
#else
#define CORO_KEYWORD(NAME) _Static_assert(__is_identifier(NAME), #NAME)
#endif

CORO_KEYWORD(co_await);
CORO_KEYWORD(co_return);
CORO_KEYWORD(co_yield);
