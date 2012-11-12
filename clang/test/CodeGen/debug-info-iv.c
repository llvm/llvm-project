// RUN: %clang_cc1 -triple x86_64-apple-darwin -Os -S -g  -o - %s | FileCheck %s
// REQUIRES: x86-registered-target

int calculate(int);
static void test_indvars(int *Array1, int Array2[100][200]) {
  unsigned i, j;
  Array1[1] = Array2[3][6] = 12345;

  for (i = 0; i < 100; i+=2)
    Array1[i] = i;           /* Step by non unit amount */

  for (i = 3; i < 103; i++)
    Array1[i] = i+4;         /* Step with an offset */

  for (i = 13; i < 100; i++)
    for (j = 0; j < 100; j+=3)       /* 2d array access */
      Array2[i][j/3] = Array2[i][i];
}


int main() {
  int Array[100][200], i, j;
  double sum = 0.0;

  for (i=0; i < 100; i+=2)
    for (j=0; j < 200; j++)
      Array[i][j] = 0;
  test_indvars(Array[0], Array);

//CHECK:	.loc	2 31
  for (i=0; i < 100; i+=2)
    for (j=0; j < 200; j++)
      sum += Array[i][j];

  return calculate(sum);
}
