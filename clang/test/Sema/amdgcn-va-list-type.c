// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple spirv64-amd-amdhsa -fsyntax-only -verify

// expected-no-diagnostics

typedef char* va_list;

void foo(const char* f, ...) {
    int r;
    va_list args;
    __builtin_va_start(args, f);
    __builtin_va_end(args);
}