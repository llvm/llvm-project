#include <cstdio>
#include <iostream>
using namespace std;

#include <Kokkos_Core.hpp>

typedef Kokkos::View<double*> View;

const size_t N = 16;
const size_t M = 16;

int main (int argc, char* argv[]) {

  Kokkos::initialize (argc, argv);

  { // scope used for cleanup (to hush kokkos)... 
    double * const y = new double[N];
    double * const x = new double[M];
    double * const A = new double[N * M];

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
      y[i] = 1;
    });

    Kokkos::parallel_for(M, KOKKOS_LAMBDA(int i) {
      x[i] = 1;
    });

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int j) {
      for(int i = 0; i < M; ++i) {
        A[j * M + i] = j;
      }
    });

    cout << "y: ";
    for(int i = 0; i < N; i++) {
      cout << y[i] << " ";
    }
    cout << endl;

    cout << "x: ";
    for(int i = 0; i < M; i++) {
      cout << x[i] << " ";
    }
    cout << endl;

    cout << "A: ";
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < M; j++) {
        cout << A[j * M + i] << " ";
      }
      cout << endl;
      cout << "   ";
    }
    cout << endl;
  }

  Kokkos::finalize ();
  return 0;
}
