// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %run %t

#include <windows.h>

typedef struct _S {
  unsigned int bf1:1;
  unsigned int bf2:2;
  unsigned int bf3:3;
  unsigned int bf4:4;
} S;

int main(void) {
  S *s = (S*)malloc(sizeof(S));
  s->bf1 = 1;
  s->bf2 = 2;
  s->bf3 = 3;
  s->bf4 = 4;
  free(s);
  return 0;
}
