// RUN: %clang_cc1 -std=c++03 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++03 -verify=ref,both %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both %s

/// The weak attribute is dropped after the initializer has been evaluated.
__attribute__((weak)) const unsigned int test10_bound = 10; // both-error {{weak declaration cannot have internal linkage}}
char test10_global[test10_bound];
void test10(void) {
  char test10_local[test10_bound] = "help";
}
int test10_size_check[sizeof(test10_global) == 10 ? 1 : -1];
