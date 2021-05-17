// RUN: %clang_cc1 -emit-llvm %s -o /dev/null -Wall -Werror
void bork() {
  char * volatile p = 0;
  volatile int cc = 0;
  p += cc;
}
