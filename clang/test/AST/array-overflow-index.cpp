// RUN: %clang_cc1  --std=c++17 -fexperimental-new-constant-interpreter -verify %s


constexpr int __attribute__((vector_size(4))) test_vector = {1};

// expected-error@+1 {{constexpr function never produces a constant expression}}
constexpr int get_last_element(void) {
    // expected-note@+1 {{cannot refer to element 18446744073709551615 of array of 1 element in a constant expression}}
    return test_vector[~0UL];
}
