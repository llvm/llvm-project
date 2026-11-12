#include "instrprof-dynamic-header.h"
void foo(int K) { if (K) {} }
int main(int argc, char *argv[]) {
  foo(5);
  bar<void>(1);
  a();
  b();
  return 0;
}
