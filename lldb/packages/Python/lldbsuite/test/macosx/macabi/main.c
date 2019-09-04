#include "foo.h"
int main() {
  const char *s = "Hello MacABI!";
  return foo(); // break here
}
