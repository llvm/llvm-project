#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* #define DEBUG */

typedef struct {
  int *a1,*a2,*re,*ex,len;
} data;


void print_arr(int *a, int len) {
  printf("[%d", a[0]);
  for (int i=1; i<len; i++)
    printf(", %d", a[i]);
  printf("]");
  return;
}

data gen_data(int len) {
  data d;
  d.a1 = malloc(len*sizeof(int));
  d.a2 = malloc(len*sizeof(int));
  d.ex = malloc(len*sizeof(int));
  d.re = malloc(len*sizeof(int));
  d.len = len;

  for (int i = 0; i < len; i++) {
    int a1v = rand()%100;
    int a2v = rand()%10;
    d.a1[i] = a1v;
    d.a2[i] = a2v;
    d.ex[i] = 0;
    d.re[i] = 0;
  }
#ifdef DEBUG
  printf("a1="); print_arr(d.a1, d.len); printf("\n");
  printf("a2="); print_arr(d.a2, d.len); printf("\n");
#endif
  return d;
}

void print_cmp(data d) {
#ifdef DEBUG
  printf("ex="); print_arr(d.ex, d.len); printf("\n");
  printf("re="); print_arr(d.re, d.len); printf("\n");
#endif
}

int check_re(data d) {
  print_cmp(d);
  for (int i = 0; i < d.len; i++) {
    if (d.re[i] != d.ex[i])
      return 0;
  }
  return 1;
}
void reset_re(data d) {
  for (int i = 0; i < d.len; i++) {
    d.re[i] = 0;
  }
}

void _QPf1dc(int*, int*, int*, int*);
void _QPf1dv(int*, int*, int*, int*);
int test_1d(void (*f)(int*, int*, int*, int*), data d) {
  reset_re(d);
  f(d.a1, d.a2, d.re, &(d.len));
  for (int i = 0; i < d.len; i++) {
    d.ex[i] = d.a1[i]*2*d.a2[i];
  }
  return check_re(d);
}

void _QPf2dc(int*, int*, int*, int*, int*);
void _QPf2dv(int*, int*, int*, int*, int*);
int test_2d(void (*f)(int*,int*,int*,int*,int*), data d, int s1, int s2) {
  assert(s1*s2 == d.len);
  reset_re(d);
  f(d.a1, d.a2, d.re, &s1, &s2);
  for (int i = 0; i < s1; i++) {
    for (int j = 0; j < s2; j++) {
      int rm = j + s2*i;
      int cm = i + s1*j;
      d.ex[cm] = d.a1[cm] + d.a1[rm]*100 + d.a2[cm]*10000;
    }
  }
  return check_re(d);
}

void _QPf3dc(int*, int*, int*, int*, int*, int*);
void _QPf3dv(int*, int*, int*, int*, int*, int*);
int test_3d(void (*f)(int*,int*,int*,int*,int*,int*), data d, int s1, int s2, int s3) {
  assert(s1*s2*s3 == d.len);
  reset_re(d);
  f(d.a1, d.a2, d.re, &s1, &s2, &s3);
  for (int i = 0; i < s1; i++) {
    for (int j = 0; j < s2; j++) {
      for (int k = 0; k < s3; k++) {
        int rm = k + s3*(j + s2*i);
        int cm = i + s1*(j + s2*k);
        d.ex[cm] = d.a1[cm] + d.a1[rm]*100 + d.a2[cm]*10000;
      }
    }
  }
  return check_re(d);
}

int main() {

  data d = gen_data(60);
  printf("f1dc: %s\n", test_1d(_QPf1dc, d) ? "success" : "fail");
  printf("f1dv: %s\n", test_1d(_QPf1dv, d) ? "success" : "fail");

  d = gen_data(9);
  printf("f2dc: %s\n", test_2d(_QPf2dc, d, 3, 3) ? "success" : "fail");

  d = gen_data(100);
  printf("f2dv: %s\n", test_2d(_QPf2dv, d, 10, 10) ? "success" : "fail");


  d = gen_data(27);
  printf("f3dc: %s\n", test_3d(_QPf3dc, d, 3, 3, 3) ? "success" : "fail");
  printf("f3dv: %s\n", test_3d(_QPf3dv, d, 3, 3, 3) ? "success" : "fail");

  return 0;
}
