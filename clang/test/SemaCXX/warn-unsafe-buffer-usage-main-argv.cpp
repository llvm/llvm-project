// RUN: %clang_cc1 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify=expected,both %s
// RUN: %clang_cc1 -Wunsafe-buffer-usage -Wno-unsafe-buffer-usage-main-argv -fsafe-buffer-usage-suggestions -verify=ignored,both %s

int main(int argc, char **argv) {
  // expected-warning@+1{{unsafe buffer access}}
  char c = argv[1][0];

  int *p = nullptr;
  // both-warning@+1{{unsafe buffer access}}
  return p[5];
}

int other_func(int argc, char **argv) {
  // both-warning@+1{{unsafe buffer access}}
  return argv[1][0];
}
