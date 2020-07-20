// RUN: %clang_cc1 -triple %ms_abi_triple -fms-extensions -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple %ms_abi_triple -fms-extensions -verify -std=c++11 -fsyntax-only -x c++ %s

// Function definition.
__declspec(guard(nocf)) void testGuardNoCF() { // no warning
}

// Can not be used on variable, parameter, or function pointer declarations.
int __declspec(guard(nocf)) i;                                      // expected-warning {{'guard' attribute only applies to functions}}
void testGuardNoCFFuncParam(double __declspec(guard(nocf)) i) {}    // expected-warning {{'guard' attribute only applies to functions}}
__declspec(guard(nocf)) typedef void (*FuncPtrWithGuardNoCF)(void); // expected-warning {{'guard' attribute only applies to functions}}

// 'guard' Attribute requries an argument.
__declspec(guard) void testGuardNoCFParams() { // expected-error {{'guard' attribute takes one argument}}
}

// 'guard' Attribute requries an identifier as argument.
__declspec(guard(1)) void testGuardNoCFParamType() { // expected-error {{'guard' attribute requires an identifier}}
}

// 'guard' Attribute only takes a single argument.
__declspec(guard(nocf, nocf)) void testGuardNoCFTooManyParams() { // expected-error {{use of undeclared identifier 'nocf'}}
}

// 'guard' Attribute argument must be a supported identifier.
__declspec(guard(cf)) void testGuardNoCFInvalidParam() { // expected-warning {{'guard' attribute argument not supported: 'cf'}}
}
