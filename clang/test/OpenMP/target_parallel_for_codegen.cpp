// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

long long get_val() { return 0; }
double *g_ptr;

// CHECK-LABEL: define {{.*void}} @{{.*}}simple{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void simple(float *a, float *b, float *c, float *d) {
  // CHECK: store i32 3, i32* %
  // CHECK: icmp slt i32 %{{.+}}, 32
  // CHECK: fmul float
  // CHECK: fmul float
  // CHECK: add nsw i32 %{{.+}}, 5
  #pragma omp target parallel for device((int)*a)
  for (int i = 3; i < 32; i += 5) {
    a[i] = b[i] * c[i] * d[i];
  }

  // CHECK: call i{{.+}} @{{.+}}get_val{{.+}}()
  // CHECK: store i32 10, i32* %
  // CHECK: icmp sgt i32 %{{.+}}, 1
  // CHECK: fadd float %{{.+}}, 1.000000e+00
  // CHECK: add nsw {{.+}} %{{.+}}, 3
  // CHECK: add nsw i32 %{{.+}}, -1
  long long k = get_val();
  #pragma omp target parallel for linear(k : 3) schedule(dynamic)
  for (int i = 10; i > 1; i--) {
    a[k]++;
    k = k + 3;
  }

  // CHECK: store i32 12, i32* %
  // CHECK: store i{{.+}} 2000, i{{.+}}* %
  // CHECK: icmp uge i{{.+}} %{{.+}}, 600
  // CHECK: store double 0.000000e+00,
  // CHECK: fadd float %{{.+}}, 1.000000e+00
  // CHECK: sub i{{.+}} %{{.+}}, 400
  int lin = 12;
  #pragma omp target parallel for linear(lin : get_val()), linear(g_ptr)
  for (unsigned long long it = 2000; it >= 600; it-=400) {
    *g_ptr++ = 0.0;
    a[it + lin]++;
  }

  // CHECK: store i{{.+}} 6, i{{.+}}* %
  // CHECK: icmp sle i{{.+}} %{{.+}}, 20
  // CHECK: sub nsw i{{.+}} %{{.+}}, -4
  #pragma omp target parallel for
  for (short it = 6; it <= 20; it-=-4) {
  }

  // CHECK: store i8 122, i8* %
  // CHECK: icmp sge i32 %{{.+}}, 97
  // CHECK: add nsw i32 %{{.+}}, -1
  #pragma omp target parallel for
  for (unsigned char it = 'z'; it >= 'a'; it+=-1) {
  }

  // CHECK: store i32 100, i32* %
  // CHECK: icmp ult i32 %{{.+}}, 10
  // CHECK: add i32 %{{.+}}, 10
  #pragma omp target parallel for
  for (unsigned i=100; i<10; i+=10) {
  }

  int A;
  {
  A = -1;
  // CHECK: store i{{.+}} -10, i{{.+}}* %
  // CHECK: icmp slt i{{.+}} %{{.+}}, 10
  // CHECK: add nsw i{{.+}} %{{.+}}, 3
  #pragma omp target parallel for lastprivate(A)
  for (long long i = -10; i < 10; i += 3) {
    A = i;
  }
  }
  int R;
  {
  R = -1;
  // CHECK: store i{{.+}} -10, i{{.+}}* %
  // CHECK: icmp slt i{{.+}} %{{.+}}, 10
  // CHECK: add nsw i{{.+}} %{{.+}}, 3
  #pragma omp target parallel for reduction(*:R)
  for (long long i = -10; i < 10; i += 3) {
    R *= i;
  }
  }
}

template <class T, unsigned K> T tfoo(T a) { return a + K; }

template <typename T, unsigned N>
int templ1(T a, T *z) {
  #pragma omp target parallel for collapse(N)
  for (int i = 0; i < N * 2; i++) {
    for (long long j = 0; j < (N + N + N + N); j += 2) {
      z[i + j] = a + tfoo<T, N>(i + j);
    }
  }
  return 0;
}

// Instatiation templ1<float,2>
// CHECK-LABEL: define {{.*i32}} @{{.*}}templ1{{.*}}(float {{.+}}, float* {{.+}})
void inst_templ1() {
  float a;
  float z[100];
  templ1<float,2> (a, z);
}


typedef int MyIdx;

class IterDouble {
  double *Ptr;
public:
  IterDouble operator++ () const {
    IterDouble n;
    n.Ptr = Ptr + 1;
    return n;
  }
  bool operator < (const IterDouble &that) const {
    return Ptr < that.Ptr;
  }
  double & operator *() const {
    return *Ptr;
  }
  MyIdx operator - (const IterDouble &that) const {
    return (MyIdx) (Ptr - that.Ptr);
  }
  IterDouble operator + (int Delta) {
    IterDouble re;
    re.Ptr = Ptr + Delta;
    return re;
  }

  ///~IterDouble() {}
};

// CHECK-LABEL: define {{.*void}} @{{.*}}iter_simple{{.*}}
void iter_simple(IterDouble ia, IterDouble ib, IterDouble ic) {
//
// Calculate number of iterations before the loop body.
// CHECK: invoke {{.*}}i32 @{{.*}}IterDouble{{.*}}
  #pragma omp target parallel for
  for (IterDouble i = ia; i < ib; ++i) {
// Call of operator+ (i, IV).
// CHECK: {{%.+}} = invoke {{.+}} @{{.*}}IterDouble{{.*}}
// ... loop body ...
   *i = *ic * 0.5;
// Float multiply and save result.
// CHECK: [[MULR:%.+]] = fmul double {{%.+}}, 5.000000e-01
// CHECK-NEXT: invoke {{.+}} @{{.*}}IterDouble{{.*}}
// CHECK: store double [[MULR:%.+]], double* [[RESULT_ADDR:%.+]]
   ++ic;
  }
// CHECK: ret void
}


// CHECK-LABEL: define {{.*void}} @{{.*}}collapsed{{.*}}
void collapsed(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  unsigned j; // middle loop couter, leads to unsigned icmp in loop header.
  // k declared in the loop init below
  short l; // inner loop counter
//
  #pragma omp target parallel for collapse(4)
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 2u; j < 5u; j++) //3 iterations
      for (int k = 3; k <= 6; k++) // 4 iterations
        for (l = 4; l < 9; ++l) // 5 iterations
        {
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]
    float res = b[j] * c[k];
    a[i] = res * d[l];
  }
// CHECK: ret void
}

extern char foo();
extern double globalfloat;

// CHECK-LABEL: define {{.*void}} @{{.*}}widened{{.*}}
void widened(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  short j; // inner loop counter
  globalfloat = 1.0;
  int localint = 1;
// CHECK: store double {{.+}}, double* [[GLOBALFLOAT:@.+]],
  #pragma omp target parallel for collapse(2) private(globalfloat, localint)
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 0; j < foo(); j++) // foo() iterations
  {
// ... loop body ...
//
// Here we expect store into private double var, not global
// CHECK: store double {{.+}}, double* [[GLOBALFLOAT]]
    globalfloat = (float)j/i;
    float res = b[j] * c[j];
// Store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]
    a[i] = res * d[i];
// Then there's a store into private var localint:
// CHECK: store i32 {{.+}}, i32* [[LOCALINT:%[^,]+]]
    localint = (int)j;
  }
//
// Here we expect store into original localint, not its privatized version.
// CHECK: store i32 {{.+}}, i32* [[LOCALINT]]
  localint = (int)j;
// CHECK: ret void
}

// TERM_DEBUG-LABEL: bar
int bar() {return 0;};

// TERM_DEBUG-LABEL: parallel_simd
void parallel_simd(float *a) {
#pragma omp target parallel for
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += bar();
}
#endif // HEADER

