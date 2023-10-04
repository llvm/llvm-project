// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fexperimental-decimal-floating-point -fsyntax-only -verify=c %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -std=c++2c -fexperimental-decimal-floating-point -fsyntax-only -verify=cxx %s

// c-no-diagnostics

// _Decimal32, _Decimal64, and _Decimal128 are never keywords in C++.
_Decimal32 d32; // cxx-error {{unknown type name '_Decimal32'}}
_Decimal64 d64; // cxx-error {{unknown type name '_Decimal64'}}
_Decimal128 d28; // cxx-error {{unknown type name '_Decimal128'}}
