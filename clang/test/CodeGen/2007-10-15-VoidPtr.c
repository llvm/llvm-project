// RUN: %clang_cc1 -emit-llvm %s -o /dev/null
void bork(void **data) {
  (*(unsigned short *) (&(data[37])[927]) = 0);
}
