// RUN: %clang_cc1 -std=c++17 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify=expected-17,both-17 %s
// RUN: %clang_cc1 -std=c++17 -Wunsafe-buffer-usage -Wno-unsafe-buffer-usage-main-argv -fsafe-buffer-usage-suggestions -verify=both-17 %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify=expected-20,both-20 %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -Wno-unsafe-buffer-usage-main-argv -fsafe-buffer-usage-suggestions -verify=both-20 %s

// expected-20-warning@+1{{unsafe pointer used for buffer access}}
int main(int argc, char **argv) {
  // expected-17-warning@+2{{unsafe buffer access}}
  // expected-20-note@+1{{used in buffer access here}}
  char c = argv[1][0];

  // expected-17-warning@+2{{unsafe pointer arithmetic}}
  // expected-20-note@+1{{used in pointer arithmetic here}}
  char **ptr1 = argv + 1;
  // expected-17-warning@+2{{unsafe pointer arithmetic}}
  // expected-20-note@+1{{used in pointer arithmetic here}}
  argv++;
  // expected-17-warning@+2{{unsafe pointer arithmetic}}
  // expected-20-note@+1{{used in pointer arithmetic here}}
  argv += 1;

  // both-20-warning@+2{{unsafe pointer used for buffer access}}
  // both-20-note@+1{{change type of 'p' to 'std::span' to preserve bounds information}}
  int *p = nullptr;
  // both-17-warning@+2{{unsafe buffer access}}
  // both-20-note@+1{{used in buffer access here}}
  return p[5];
}

// both-20-warning@+2{{unsafe pointer used for buffer access}}
// both-20-note@+1{{change type of 'argv' to 'std::span' to preserve bounds information}}
int other_func(int argc, char **argv) {
  // both-17-warning@+2{{unsafe buffer access}}
  // both-20-note@+1{{used in buffer access here}}
  return argv[1][0];
}
