// RUN: %clang_cc1 -triple riscv32-unknown-elf -target-abi ilp32e -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64-unknown-elf -target-feature +e -target-abi lp64e -fsyntax-only -verify %s

// The Swift calling convention is not supported on the E ABIs: the Swift
// context register (x20) does not exist under the reduced register set.

#if __has_extension(swiftcc)
#error swiftcc should not be available on the E ABIs
#endif

#if __has_extension(swiftasynccc)
#error swiftasynccc should not be available on RISC-V
#endif

// expected-error@+1 {{'swiftcall' calling convention is not supported for this target}}
void __attribute__((swiftcall)) f(void *__attribute__((swift_context)) ctx) {}
