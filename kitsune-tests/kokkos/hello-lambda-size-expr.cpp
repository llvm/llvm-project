// A modified version of the simple lambda tests in hello-lambda.cpp 
// that uses expressions for the size (trip count) for the parallel_for 
// constructs. 
#include <cstdio>
#include <Kokkos_Core.hpp>

const float QUARTER_NTRIPS = 2.5;

unsigned int check_ntrips(unsigned int ntrips) {
  if (ntrips != 10) 
    return 10;
  else 
    return ntrips;
}

int main (int argc, char* argv[]) {

  Kokkos::initialize (argc, argv);

  {
    Kokkos::parallel_for(2*4 + 2, KOKKOS_LAMBDA(const int i) {
      std::printf("hello from %i\n", i);
    });

    printf("\n"); 
    
    Kokkos::parallel_for(check_ntrips(100), KOKKOS_LAMBDA(const int i) {
      std::printf("hello from %i\n", i);
    });

    printf("\n"); 

    Kokkos::parallel_for(QUARTER_NTRIPS * 4, KOKKOS_LAMBDA(const int i) {
      std::printf("hello from %i\n", i);
    });

  }
  Kokkos::finalize ();
  return 0;
}
