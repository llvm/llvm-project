
#include "plist-macros-ctu.h"

#define M *X = (int *)0

void F1(int **X) {
  M;
}

#undef M
#define M *Y = (int *)0

void F2(int **Y) {
  M;
}

#define M1 *Z = (int *)0

void F3(int **Z) {
  M1;
}
