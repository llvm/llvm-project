// RUN: %clang_cc1  --std=c++17  -verify=none %s
// RUN: %clang_cc1  --std=c++17 -fexperimental-new-constant-interpreter -verify=experiment %s

// none-no-diagnostics

constexpr int __attribute__((vector_size(4))) test_vector = {1};

constexpr int get_last_element(void) { // experiment-error {{constexpr function never produces a constant expression}}
    return test_vector[~0UL]; // experiment-note {{cannot refer to element}}
}
