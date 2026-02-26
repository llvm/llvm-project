// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple armv7-linux-gnueabihf %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple riscv32-unknown-elf %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple mipsel-linux-gnu %s

// Verify __int256 is rejected on 32-bit targets.
// On 32-bit, the __int256_t typedef is not predefined, so use the keyword.

__int256 x; // expected-error {{__int256 is not supported on this target}}
unsigned __int256 y; // expected-error {{__int256 is not supported on this target}}

void f(__int256 a) {} // expected-error {{__int256 is not supported on this target}}
__int256 g(void); // expected-error {{__int256 is not supported on this target}}
