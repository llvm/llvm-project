// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple mipsel-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple mips64-unknown-linux-gnuabi64 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux-gnuabi64 -fsyntax-only -verify %s

// swiftcall is supported on MIPS; swiftasynccall is not, because the backend
// has no guaranteed tail call support for it.

void __attribute__((swiftcall)) f(void *__attribute__((swift_context)) ctx) {}

#if !__has_extension(swiftcc)
#error swiftcc should be available on MIPS
#endif

#if __has_extension(swiftasynccc)
#error swiftasynccc should not be available on MIPS
#endif

// expected-error@+1 {{'swiftasynccall' calling convention is not supported for this target}}
void __attribute__((swiftasynccall)) g(void *__attribute__((swift_async_context)) ctx) {}
