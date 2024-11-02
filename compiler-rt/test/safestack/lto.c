// REQUIRES: lto

// RUN: %clang_lto_safestack %s -o %t
// RUN: %run %t

// Test that safe stack works with LTO.
int puts(const char *);

int main() {
  char c[] = "hello world";
  puts(c);
  return 0;
}
