// Helper binary for
// lit_tests/TestCases/Darwin/unset-insert-libraries-on-exec.cpp
// Prints the environment variable with the given name.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s ENVNAME\n", argv[0]);
    exit(1);
  }
  const char *value = getenv(argv[1]);
  if (value) {
    printf("%s = %s\n", argv[1], value);
  } else {
    printf("%s not set.\n", argv[1]);
  }
  return 0;
}
