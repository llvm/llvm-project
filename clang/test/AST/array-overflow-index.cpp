// RUN: %clang_cc1  --std=c++17  -fexperimental-new-constant-interpreter -verify=experiment %s
// RUN: %clang_cc1  --std=c++17  -triple x86_64-pc-win32 -verify=experiment %s
// RUN: %clang_cc1  --std=c++17  -triple x86_64-pc-linux -verify=none %s

#ifndef _MSC_VER
// none-no-diagnostics
#endif

constexpr int __attribute__((vector_size(4))) test_vector = {1};
constexpr int get_last_element(void) { // experiment-error {{constexpr function never produces a constant expression}}
    return test_vector[~0UL]; // experiment-note {{cannot refer to element}}
}

