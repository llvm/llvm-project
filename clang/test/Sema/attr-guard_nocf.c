// RUN: %clang_cc1 -triple %ms_abi_triple -fms-extensions -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple %ms_abi_triple -fms-extensions -verify -std=c++11 -fsyntax-only -x c++ %s
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -verify -std=c++11 -fsyntax-only -x c++ %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -verify -std=c++11 -fsyntax-only -x c++ %s

// The x86_64-w64-windows-gnu version tests mingw target, which relies on
// __declspec(...) being defined as __attribute__((...)) by compiler built-in.

#if defined(_WIN32)

// Function definition.
__declspec(guard(nocf)) void testGuardNoCF(void) { // no warning
}

// Function definition using GNU-style attribute without relying on the
// __declspec define for mingw.
__attribute__((guard(nocf))) void testGNUStyleGuardNoCF(void) { // no warning
}

// Can not be used on variable, parameter, or function pointer declarations.
int __declspec(guard(nocf)) i;                                      // expected-warning {{'guard' attribute only applies to functions}}
void testGuardNoCFFuncParam(double __declspec(guard(nocf)) i) {}    // expected-warning {{'guard' attribute only applies to functions}}
__declspec(guard(nocf)) typedef void (*FuncPtrWithGuardNoCF)(void); // expected-warning {{'guard' attribute only applies to functions}}

// 'guard' Attribute requries an argument.
__declspec(guard) void testGuardNoCFParams(void) { // expected-error {{'guard' attribute takes one argument}}
}

// 'guard' Attribute requries an identifier as argument.
__declspec(guard(1)) void testGuardNoCFParamType(void) { // expected-error {{'guard' attribute requires an identifier}}
}

// 'guard' Attribute only takes a single argument.
__declspec(guard(nocf, nocf)) void testGuardNoCFTooManyParams(void) { // expected-error {{use of undeclared identifier 'nocf'}}
}

// 'guard' Attribute argument must be a supported identifier.
__declspec(guard(cf)) void testGuardNoCFInvalidParam(void) { // expected-warning {{'guard' attribute argument not supported: 'cf'}}
}

#else

__attribute((guard(nocf))) void testGNUStyleGuardNoCF(void) {} // expected-warning {{unknown attribute 'guard' ignored}}

#endif
