// XFAIL: aix
// This test checks that the cycle detection algorithm in llvm-cov is able to
// handle complex block graphs by skipping zero count cycles.
//
// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clang --coverage %s -o %t
// RUN: rm -f gcov-complex-line.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-complex-line.c | FileCheck %s

#define M_0     \
  do {          \
    if (x == 0) \
      x = 0;    \
    else        \
      x = 1;    \
  } while (0)
#define M_1 \
  do {      \
    M_0;    \
    M_0;    \
    M_0;    \
    M_0;    \
  } while (0)
#define M_2 \
  do {      \
    M_1;    \
    M_1;    \
    M_1;    \
    M_1;    \
  } while (0)
#define M_3 \
  do {      \
    M_2;    \
    M_2;    \
    M_2;    \
    M_2;    \
  } while (0)
#define M_4 \
  do {      \
    M_3;    \
    M_3;    \
    M_3;    \
    M_3;    \
  } while (0)
#define COMPLEX_LINE              \
  do {                            \
    for (int i = 0; i < 100; ++i) \
      M_4;                        \
  } while (0)

int main() {
  volatile int x = 0;
  // In the following line, the number of cycles in the block graph is at least
  // 2^256, where 256 is the number of expansions of M_0.
  COMPLEX_LINE; // CHECK: 101: [[#@LINE]]: COMPLEX_LINE;
}
