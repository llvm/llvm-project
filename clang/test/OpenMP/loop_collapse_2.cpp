// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -verify %s

// We just want to try out a range for statement... this seems a bit OTT.
template<typename T>
class fakevector {
  T *contents;
  long size;
public:
  fakevector(long sz) : size(sz) {
    contents = new T[sz];
  }
  ~fakevector() {
    delete[] contents;
  }
  T& operator[](long x) { return contents[x]; }
  typedef T *iterator;
  fakevector<T>::iterator begin() {
    return &contents[0];
  }
  fakevector<T>::iterator end() {
    return &contents[size];
  }
};

void func( double *A, int N, int M, int NB ) {
#pragma omp parallel
  {
    int nblks = (N-1)/NB;
    int lnb = ((N-1)/NB)*NB;
#pragma omp for collapse(2)
    for (int jblk = 0 ; jblk < nblks ; jblk++ ) {
      int jb = (jblk == nblks - 1 ? lnb : NB);
      for (int jk = 0; jk < N; jk+=jb) {  // expected-error{{cannot use variable 'jb' in collapsed imperfectly-nested loop increment statement}}
      }
    }

#pragma omp for collapse(2)
    for (int a = 0; a < N; a++) {
        for (int b = 0; b < M; b++) {
          int cx = a+b < NB ? a : b;
          for (int c = 0; c < cx; c++) {
          }
        }
    }

    fakevector<float> myvec{N};
#pragma omp for collapse(2)
    for (auto &a : myvec) {
      fakevector<float> myvec3{M};
      for (auto &b : myvec3) {  // expected-error{{cannot use variable 'myvec3' in collapsed imperfectly-nested loop init statement}}
      }
    }

    fakevector<float> myvec2{M};

#pragma omp for collapse(3)
    for (auto &a : myvec) {
      for (auto &b : myvec2) {
        int cx = a < b ? N : M;
        for (int c = 0; c < cx; c++) {  // expected-error {{cannot use variable 'cx' in collapsed imperfectly-nested loop condition statement}}
        }
      }
    }

#pragma omp for collapse(3)
    for (auto &a : myvec) {
      int cx = a < 5 ? M : N;
      for (auto &b : myvec2) {
        for (int c = 0; c < cx; c++) {  // expected-error{{cannot use variable 'cx' in collapsed imperfectly-nested loop condition statement}}
        }
      }
    }
  }
}

int main(void) {
  double arr[256];
  func (arr, 16, 16, 16);
  return 0;
}
