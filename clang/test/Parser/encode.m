// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

int main(void) {
  const char ch = @encode(char *)[0];
  char c = @encode(char *)[0] + 4;
  return c;
}

