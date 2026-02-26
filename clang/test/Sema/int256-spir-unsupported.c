// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64-unknown-unknown %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv32-unknown-unknown %s

// Verify __int256 is rejected on SPIR targets (even 64-bit SPIR).
// On SPIR, the __int256_t typedef is not predefined, so use the keyword.

__int256 x; // expected-error {{__int256 is not supported on this target}}
unsigned __int256 y; // expected-error {{__int256 is not supported on this target}}
