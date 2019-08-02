// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

// tdestroy is a GNU extension
// UNSUPPORTED: netbsd, freebsd

#include <assert.h>
#include <search.h>
#include <stdlib.h>

int compare(const void *pa, const void *pb) {
  int a = *(const int *)pa;
  int b = *(const int *)pb;
  if (a < b)
    return -1;
  else if (a > b)
    return 1;
  else
    return 0;
}

void myfreenode(void *p) {
  delete (int *)p;
}

int main(void) {
  void *root = NULL;
  for (int i = 0; i < 5; ++i) {
    int *p = new int(i);
    void *q = tsearch(p, &root, compare);
    if (q == NULL)
      exit(1);
    if (*(int **)q != p)
      delete p;
  }

  tdestroy(root, myfreenode);

  return 0;
}
