// RUN: clang-tidy %s -checks=-*,modernize-loop-convert -- -std=c11 | count 0

// Note: this test expects no diagnostics, but FileCheck cannot handle that,
// hence the use of | count 0.

int arr[6] = {1, 2, 3, 4, 5, 6};

void f(void) {
  for (int i = 0; i < 6; ++i) {
    (void)arr[i];
  }
}
