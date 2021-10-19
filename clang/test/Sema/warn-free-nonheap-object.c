// RUN: %clang_cc1 -Wfree-nonheap-object -fsyntax-only -verify %s

typedef __SIZE_TYPE__ size_t;
void *malloc(size_t);
void free(void *);

struct S {
  int I;
  char *P;
};

int GI;
void test() {
  {
    free(&GI); // expected-warning {{attempt to call free on non-heap object 'GI'}}
  }
  {
    static int SI = 0;
    free(&SI); // expected-warning {{attempt to call free on non-heap object 'SI'}}
  }
  {
    int I = 0;
    free(&I); // expected-warning {{attempt to call free on non-heap object 'I'}}
  }
  {
    int I = 0;
    int *P = &I;
    free(P); // FIXME diagnosing this would require control flow analysis.
  }
  {
    void *P = malloc(8);
    free(P);
  }
  {
    int A[] = {0, 1, 2, 3};
    free(A);  // expected-warning {{attempt to call free on non-heap object 'A'}}
    free(&A); // expected-warning {{attempt to call free on non-heap object 'A'}}
  }
  {
    struct S s;
    free(&s.I); // expected-warning {{attempt to call free on non-heap object 'I'}}
    free(s.P);
  }
}
