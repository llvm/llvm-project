// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++17 -Wno-vla -fsyntax-only -verify -fptrauth-intrinsics %s

int n;
constexpr bool compare_result = __builtin_ptrauth_sign_constant(&n, 2, 0) == &n;
// expected-error@-1 {{constant expression}}
// expected-note@-2 {{comparison against opaque constant address '&__builtin_ptrauth_sign_constant(&n, 2, 0)'}}