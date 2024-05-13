// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -fcxx-exceptions  -fsafe-buffer-usage-suggestions -verify %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fcxx-exceptions -fdiagnostics-parseable-fixits  -fsafe-buffer-usage-suggestions %s 2>&1 | FileCheck %s

// We do not fix parameters of the `main` function

// CHECK-NOT: fix-it:

// main function
int main(int argc, char *argv[]) { // expected-warning{{'argv' is an unsafe pointer used for buffer access}}
  char tmp;
  tmp = argv[5][5];                // expected-note{{used in buffer access here}} \
				      expected-warning{{unsafe buffer access}}
}
