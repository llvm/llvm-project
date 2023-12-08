// Test identical code folding handling.
#include <stdio.h>

int foo() {
  return 0;
}

int bar() {
  return 0;
}

int main() {
  return foo();
}
