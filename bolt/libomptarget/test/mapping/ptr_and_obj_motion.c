// RUN: %libomptarget-compile-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-run-and-check-nvptx64-nvidia-cuda

#include <stdio.h>

typedef struct {
  double *dataptr;
  int dummy1;
  int dummy2;
} DV;

void init(double vertexx[]) {
  #pragma omp target map(vertexx[0:100])
  {
    printf("In init: %lf, expected 100.0\n", vertexx[77]);
    vertexx[77] = 77.0;
  }
}

void change(DV *dvptr) {
  #pragma omp target map(dvptr->dataptr[0:100])
  {
    printf("In change: %lf, expected 77.0\n", dvptr->dataptr[77]);
    dvptr->dataptr[77] += 1.0;
  }
}

int main() {
  double vertexx[100];
  vertexx[77] = 100.0;

  DV dv;
  dv.dataptr = &vertexx[0];

  #pragma omp target enter data map(to:vertexx[0:100])

  init(vertexx);
  change(&dv);

  #pragma omp target exit data map(from:vertexx[0:100])

  // CHECK: Final: 78.0
  printf("Final: %lf\n", vertexx[77]);
}

