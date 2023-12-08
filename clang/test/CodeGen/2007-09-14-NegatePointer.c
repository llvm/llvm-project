// RUN: %clang_cc1 -emit-llvm %s -o - 
// PR1662

int foo(unsigned char *test) {
  return 0U - (unsigned int )test;
}

