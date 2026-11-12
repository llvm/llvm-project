// RUN: %clang_cc1 -emit-llvm %s -o /dev/null -Wall -Wno-unused-but-set-variable -Werror
void bork(void) {
  char * volatile p = 0;
  volatile int cc = 0;
  p += cc;
}
