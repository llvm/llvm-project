// RUN: %clang_cc1 -triple riscv32-unknown-elf -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -fsyntax-only -verify %s

// swiftcall is supported on RISC-V; swiftasynccall is not, because lowering
// it needs guaranteed tail calls the backend does not provide.

void __attribute__((swiftcall)) f(void *__attribute__((swift_context)) ctx) {}

#if !__has_extension(swiftcc)
#error swiftcc should be available on RISC-V
#endif

#if __has_extension(swiftasynccc)
#error swiftasynccc should not be available on RISC-V
#endif

// expected-error@+1 {{'swiftasynccall' calling convention is not supported for this target}}
void __attribute__((swiftasynccall)) g(void *__attribute__((swift_async_context)) ctx) {}
