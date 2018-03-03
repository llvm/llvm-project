// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-cuda-mode -fopenmp-host-ir-file-path %t-ppc-host.bc -o -

template <typename tx, typename ty>
struct TT {
  tx X;
  ty Y;
};

int foo(int n, double *ptr) {
  int a = 0;
  short aa = 0;
  float b[10];
  double c[5][10];
  TT<long long, char> d;

#pragma omp target firstprivate(a) map(tofrom: b) // expected-note 2 {{defined as threadprivate or thread local}}
  {
    int c;                               // expected-note {{defined as threadprivate or thread local}}
#pragma omp parallel shared(a, b, c, aa) // expected-error 3 {{threadprivate or thread local variable cannot be shared}}
    b[a] = a;
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) // expected-note {{defined as threadprivate or thread local}}
#pragma omp parallel shared(i) // expected-error {{threadprivate or thread local variable cannot be shared}}
    ++i;
  }

#pragma omp target map(aa, b, c, d)
  {
    int e;                         // expected-note {{defined as threadprivate or thread local}}
#pragma omp parallel private(b, e) // expected-error {{threadprivate or thread local variable cannot be private}}
    {
      aa += 1;
      b[2] = 1.0;
      c[1][2] = 1.0;
      d.X = 1;
      d.Y = 1;
    }
  }

#pragma omp target private(ptr)
  {
    ptr[0]++;
  }

  return a;
}

template <typename tx>
tx ftemplate(int n) {
  tx a = 0;
  tx b[10];

#pragma omp target reduction(+ \
                             : a, b) // expected-note {{defined as threadprivate or thread local}}
  {
    int e;                        // expected-note {{defined as threadprivate or thread local}}
#pragma omp parallel shared(a, e) // expected-error 2 {{threadprivate or thread local variable cannot be shared}}
    a += 1;
    b[2] += 1;
  }

  return a;
}

static int fstatic(int n) {
  int a = 0;
  char aaa = 0;
  int b[10];

#pragma omp target firstprivate(a, aaa, b)
  {
    a += 1;
    aaa += 1;
    b[2] += 1;
  }

  return a;
}

struct S1 {
  double a;

  int r1(int n) {
    int b = n + 1;

#pragma omp target firstprivate(b) // expected-note {{defined as threadprivate or thread local}}
    {
      int c;                      // expected-note {{defined as threadprivate or thread local}}
#pragma omp parallel shared(b, c) // expected-error 2 {{threadprivate or thread local variable cannot be shared}}
      this->a = (double)b + 1.5;
    }

    return (int)b;
  }
};

int bar(int n, double *ptr) {
  int a = 0;
  a += foo(n, ptr);
  S1 S;
  a += S.r1(n);
  a += fstatic(n);
  a += ftemplate<int>(n); // expected-note {{in instantiation of function template specialization 'ftemplate<int>' requested here}}

  return a;
}

