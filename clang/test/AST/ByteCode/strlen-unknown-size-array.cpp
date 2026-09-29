// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -std=c++20 %s -verify=ref

// expected-no-diagnostics
// ref-no-diagnostics

/// Test that __builtin_strlen() on external/unknown declarations doesn't crash the bytecode interpreter.
extern const char s[];
void foo(char *x)
{
    unsigned long len = __builtin_strlen(s);
    __builtin_strcpy(x, s);
}
