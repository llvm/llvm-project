#include <cstdio>
#include <iostream>
using namespace std;

#include <Kokkos_Core.hpp>

typedef Kokkos::View<double*> View;

const size_t SIZE = 16;

void dump(View &v) {
  for(size_t i = 0; i < SIZE; ++i) { 
    cout << v(i) << ' ';
  }
  cout << endl;
}

void initialize(View &v) {
  Kokkos::parallel_for(SIZE, KOKKOS_LAMBDA(const int i) {
    v(i) = double(i);
  });
  dump(v);
}


int main (int argc, char* argv[]) {

  Kokkos::initialize (argc, argv);

  { // scope used for cleanup (to hush kokkos)... 
    View a("a", SIZE);
    View b("b", SIZE);
    View c("c", SIZE);

    initialize(a);
    initialize(b);

    Kokkos::parallel_for(SIZE, KOKKOS_LAMBDA (const int i) {
      c(i) = a(i) + b(i);
    });

    dump(c); 
  }

  Kokkos::finalize ();
}
