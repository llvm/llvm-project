#include <cstdio>
#include <iostream>
using namespace std;

#include <Kokkos_Core.hpp>

struct hello_world {
  KOKKOS_INLINE_FUNCTION 
  void operator() (const int i) const {
    printf("hello from %i\n", i);
  }
};

int main (int argc, char* argv[]) {

  Kokkos::initialize (argc, argv);
  {
    Kokkos::parallel_for("debug", 15, KOKKOS_LAMBDA(const int i) {
	printf("hello from %i\n", i);
      });
  }
  Kokkos::finalize ();
  return 0;
}
