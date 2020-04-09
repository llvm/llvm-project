#include <stdio.h>

void _QPsetall(float (*)[20][10], float *);
void _QPsub1(float (*)[20][10], float (*)[20][10], float (*)[20][10]);

static float arr_a[20][10];
static float arr_b[20][10];
static float arr_c[20][10];
static float x;

int main()
{
  x = 4.0;
  _QPsetall(&arr_a, &x);
  x = 5.0;
  arr_a[5][5] = 2.0;
  _QPsetall(&arr_b, &x);
  printf("sub1\n");
  _QPsub1(&arr_c, &arr_b, &arr_a);
  printf("c(1,1) = %f\n", arr_c[0][0]);
  printf("c(2,9) = %f\n", arr_c[8][1]);
  printf("c(6,6) = %f\n", arr_c[5][5]);
  return 0;
}
