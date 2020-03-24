// Very simple test of kokkos with two common forms of the 
// parallel_for construct.  We should be able to transform 
// all constructs from lambda into simple loops... 
#include <cstdio>
#include <Kokkos_Core.hpp>

const unsigned int NTIMES = 10;

int main (int argc, char* argv[]) {

  Kokkos::initialize (argc, argv);

  {
    Kokkos::parallel_for(NTIMES, KOKKOS_LAMBDA(const int i) {
	printf("hello from %i\n", i);
      });

    printf("\n"); 

    Kokkos::parallel_for("hello1", NTIMES, KOKKOS_LAMBDA(const int i) {
	printf("hello from %i\n", i);
      });
  }

  Kokkos::finalize ();
  return 0;
}
