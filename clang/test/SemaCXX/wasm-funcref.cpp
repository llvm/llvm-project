// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -fsyntax-only -verify -triple wasm32 -Wno-unused-value -target-feature +reference-types %s

// Testing that funcrefs work on template aliases
// expected-no-diagnostics

using IntIntFuncref = int(*)(int) __funcref;
using DoubleQual = IntIntFuncref __funcref;

int get(int);

IntIntFuncref getFuncref() {
    return get;
}
