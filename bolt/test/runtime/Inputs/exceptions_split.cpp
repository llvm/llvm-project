// Test that we can have a statement that throws in hot cold
// and a landing pad in cold code.
//
// Record performance data with no args. Run test with 2 args.

#include <stdint.h>
#include <stdio.h>

int foo() { return 0; }

void bar(int a) {
  if (a > 2 && a % 2)
    throw new int();
}

void filter_only() { foo(); }

int main(int argc, char **argv) {
  unsigned r = 0;

  uint64_t limit = (argc >= 2 ? 10 : 5000);
  for (uint64_t i = 0; i < limit; ++i) {
    i += foo();
    try {
      bar(argc);
      try {
        if (argc >= 2)
          throw new int();
      } catch (...) {
        printf("catch 2\n");
        throw new int();
      }
    } catch (...) {
      printf("catch 1\n");
    }
  }

  return 0;
}
