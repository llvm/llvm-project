// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -verify %s

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

#pragma omp for collapse(3)
    for (int a = 0; a < N; a++) {
      for (int b = 0; b < M; b++) {
        int cx = a+b < NB ? a : b;
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
