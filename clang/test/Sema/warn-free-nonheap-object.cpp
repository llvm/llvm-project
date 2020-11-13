// RUN: %clang_cc1 -Wfree-nonheap-object -std=c++11 -x c++ -fsyntax-only -verify %s

extern "C" void free(void *) {}

namespace std {
using size_t = decltype(sizeof(0));
void *malloc(size_t);
void free(void *p);
} // namespace std

int GI;

struct S {
  operator char *() { return ptr; }

  void CFree() {
    ::free(&ptr); // expected-warning {{attempt to call free on non-heap object 'ptr'}}
    ::free(&I);   // expected-warning {{attempt to call free on non-heap object 'I'}}
    ::free(ptr);
  }

  void CXXFree() {
    std::free(&ptr); // expected-warning {{attempt to call std::free on non-heap object 'ptr'}}
    std::free(&I);   // expected-warning {{attempt to call std::free on non-heap object 'I'}}
    std::free(ptr);
  }

private:
  char *ptr = (char *)std::malloc(10);
  static int I;
};

int S::I = 0;

void test1() {
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
    free(P);
  }
  {
    void *P = std::malloc(8);
    free(P); // FIXME diagnosing this would require control flow analysis.
  }
  {
    int A[] = {0, 1, 2, 3};
    free(A); // expected-warning {{attempt to call free on non-heap object 'A'}}
  }
  {
    int A[] = {0, 1, 2, 3};
    free(&A); // expected-warning {{attempt to call free on non-heap object 'A'}}
  }
  {
    S s;
    free(s);
    free(&s); // expected-warning {{attempt to call free on non-heap object 's'}}
  }
  {
    S s;
    s.CFree();
  }
}

void test2() {
  {
    std::free(&GI); // expected-warning {{attempt to call std::free on non-heap object 'GI'}}
  }
  {
    static int SI = 0;
    std::free(&SI); // expected-warning {{attempt to call std::free on non-heap object 'SI'}}
  }
  {
    int I = 0;
    std::free(&I); // expected-warning {{attempt to call std::free on non-heap object 'I'}}
  }
  {
    int I = 0;
    int *P = &I;
    std::free(P); // FIXME diagnosing this would require control flow analysis.
  }
  {
    void *P = std::malloc(8);
    std::free(P);
  }
  {
    int A[] = {0, 1, 2, 3};
    std::free(A); // expected-warning {{attempt to call std::free on non-heap object 'A'}}
  }
  {
    int A[] = {0, 1, 2, 3};
    std::free(&A); // expected-warning {{attempt to call std::free on non-heap object 'A'}}
  }
  {
    S s;
    std::free(s);
    std::free(&s); // expected-warning {{attempt to call std::free on non-heap object 's'}}
  }
  {
    S s;
    s.CXXFree();
  }
}
