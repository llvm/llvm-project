// RUN: %clang_cc1 -fopenmp -emit-pch -o %t.a %s
// RUN: %clang_cc1 -fopenmp -include-pch %t.a %s -ast-print -o - | FileCheck %s

// CHECK: #pragma clang loop vectorize_width(4)
// CHECK: #pragma clang loop interleave_count(8)
// CHECK: #pragma clang loop unroll_count(16){{$}}
// CHECK: #pragma clang loop vectorize(enable)
// CHECK: #pragma clang loop interleave(disable)
// CHECK: #pragma clang loop unroll(disable)
// CHECK: #pragma clang loop distribute(enable)
// CHECK: #pragma clang loop vectorize(disable)
// CHECK: #pragma clang loop interleave(enable)
// CHECK: #pragma clang loop unroll(full)
// CHECK: #pragma clang loop distribute(disable)
// FIXME: "#pragma unroll (enable)" is invalid and is not the input source.
// CHECK: #pragma unroll (enable){{$}}
// CHECK: #pragma unroll (32){{$}}
// CHECK: #pragma nounroll{{$}}
// CHECK: #pragma clang loop vectorize_width(V)
// CHECK: #pragma clang loop interleave_count(I)
// CHECK: #pragma omp simd
// CHECK: #pragma omp for
// CHECK: #pragma omp distribute

#ifndef HEADER
#define HEADER

class pragma_test {
public:
  inline void run1(int *List, int Length) {
    int i = 0;
#pragma clang loop vectorize_width(4)
#pragma clang loop interleave_count(8)
#pragma clang loop unroll_count(16)
    while (i < Length) {
      List[i] = i;
      i++;
    }
  }

  inline void run2(int *List, int Length) {
    int i = 0;
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(disable)
#pragma clang loop unroll(disable)
#pragma clang loop distribute(enable)
    while (i - 1 < Length) {
      List[i] = i;
      i++;
    }
  }

  inline void run3(int *List, int Length) {
    int i = 0;
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(enable)
#pragma clang loop unroll(full)
#pragma clang loop distribute(disable)
    while (i - 3 < Length) {
      List[i] = i;
      i++;
    }
  }

  inline void run4(int *List, int Length) {
    int i = 0;
#pragma unroll
    while (i - 3 < Length) {
      List[i] = i;
      i++;
    }
  }

  inline void run5(int *List, int Length) {
    int i = 0;
#pragma unroll 32
    while (i - 3 < Length) {
      List[i] = i;
      i++;
    }
  }

  inline void run6(int *List, int Length) {
    int i = 0;
#pragma nounroll
    while (i - 3 < Length) {
      List[i] = i;
      i++;
    }
  }

  template <int V, int I>
  inline void run7(int *List, int Length) {
#pragma clang loop vectorize_width(V)
#pragma clang loop interleave_count(I)
    for (int i = 0; i < Length; i++) {
      List[i] = i;
    }
  }

  inline void run8(int *List, int Length) {
    int i = 0;
#pragma omp loop bind(thread)
    for (int i = 0; i < Length; i++) {
      List[i] = i;
    }
  }

  inline void run9(int *List, int Length) {
    int i = 0;
#pragma omp loop bind(parallel)
    for (int i = 0; i < Length; i++) {
      List[i] = i;
    }
  }

  inline void run10(int *List, int Length) {
    int i = 0;
    int j = 0;
    #pragma omp teams
    for (int i = 0; i < Length; i++) {
      #pragma omp loop bind(teams)
      for (int j = 0; j < Length; j++) {
        List[i] = i+j;
      }
    }
  }

};
#else
void test() {
  int List[100];

  pragma_test pt;

  pt.run1(List, 100);
  pt.run2(List, 100);
  pt.run3(List, 100);
  pt.run4(List, 100);
  pt.run5(List, 100);
  pt.run6(List, 100);
  pt.run7<2, 4>(List, 100);
  pt.run8(List, 100);
  pt.run9(List, 100);
  pt.run10(List, 100);
}

#endif
