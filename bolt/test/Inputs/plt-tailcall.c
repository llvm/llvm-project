#include "stub.h"

int foo(char *c) {
  printf("");
  __attribute__((musttail)) return puts(c);
}

int main() { return foo("a"); }
