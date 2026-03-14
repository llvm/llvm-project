// RUN: %clang_cc1 -Wfree-nonheap-object -std=c++11 -x c++ -fsyntax-only -verify %s

extern "C" void free(void *) {}

namespace std {
using size_t = decltype(sizeof(0));
void *malloc(size_t);
void free(void *p);
} // namespace std

int GI;

void free_reference(char &x) { ::free(&x); }
void free_reference(char &&x) { ::free(&x); }
void std_free_reference(char &x) { std::free(&x); }
void std_free_reference(char &&x) { std::free(&x); }

struct S {
  operator char *() { return ptr1; }

  void CFree() {
    ::free(&ptr1); // expected-warning {{attempt to call free on non-heap object 'ptr1'}}
    ::free(&I);    // expected-warning {{attempt to call free on non-heap object 'I'}}
    ::free(ptr1);
    free_reference(*ptr2);
    free_reference(static_cast<char&&>(*ptr3));
  }

  void CXXFree() {
    std::free(&ptr1); // expected-warning {{attempt to call std::free on non-heap object 'ptr1'}}
    std::free(&I);    // expected-warning {{attempt to call std::free on non-heap object 'I'}}
    std::free(ptr1);
    std_free_reference(*ptr2);
    std_free_reference(static_cast<char&&>(*ptr3));
  }

private:
  char *ptr1 = (char *)std::malloc(10);
  char *ptr2 = (char *)std::malloc(10);
  char *ptr3 = (char *)std::malloc(10);
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
    char* P = (char *)std::malloc(2);
    std_free_reference(*P);
  }
  {
    char* P = (char *)std::malloc(2);
    std_free_reference(static_cast<char&&>(*P));
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
