#include <cstdio>
#include <iostream>
using namespace std;

#include <Kokkos_Core.hpp>

const size_t SIZE_1 = 0;

struct hello_world {
  KOKKOS_INLINE_FUNCTION 
  void operator() (const int i) const {
    printf("hello functor %i\n", i);
  }
};


int value() {
  return 15;
}

int main (int argc, char* argv[]) {

  Kokkos::initialize (argc, argv);
  {
    Kokkos::parallel_for(SIZE_1 + value(), KOKKOS_LAMBDA(const int i) {
	printf("hello lambda from %i\n", i);
      });

    Kokkos::parallel_for(SIZE_1 + value(), hello_world ());
  }
  Kokkos::finalize ();
  return 0;
}
