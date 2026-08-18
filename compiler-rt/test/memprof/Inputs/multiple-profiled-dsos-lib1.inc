#include <cstdlib>

extern "C" __attribute__((noinline)) void shared_allocate() {
  volatile char *P = static_cast<char *>(std::malloc(111));
  for (int I = 0; I != 111; ++I)
    P[I] = static_cast<char>(I);
  std::free(const_cast<char *>(P));
}

extern "C" __attribute__((noinline)) void lib1_entry() { shared_allocate(); }
