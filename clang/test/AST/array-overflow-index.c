// RUN: %clang_cc1 -verify %s

// expected-no-diagnostics

int __attribute__((vector_size(4))) test_vector = {1};
int get_last_element(void) {
    return test_vector[~0UL];
}
