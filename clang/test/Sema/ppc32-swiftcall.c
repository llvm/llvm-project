// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpcle-unknown-linux-gnu -fsyntax-only -verify %s

// swiftcall is supported on 32-bit PowerPC, matching the 64-bit target;
// swiftasynccall is not, for the same reason it is refused there.

void __attribute__((swiftcall)) f(void *__attribute__((swift_context)) ctx) {}

#if !__has_extension(swiftcc)
#error swiftcc should be available on 32-bit PowerPC
#endif

#if __has_extension(swiftasynccc)
#error swiftasynccc should not be available on 32-bit PowerPC
#endif

// expected-error@+1 {{'swiftasynccall' calling convention is not supported for this target}}
void __attribute__((swiftasynccall)) g(void *__attribute__((swift_async_context)) ctx) {}
