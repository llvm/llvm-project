#include <cstdlib>

extern "C" __attribute__((noinline)) void shared_allocate() {
  volatile char *P = static_cast<char *>(std::malloc(222));
  for (int I = 0; I != 222; ++I)
    P[I] = static_cast<char>(I);
  std::free(const_cast<char *>(P));
}

extern "C" __attribute__((noinline)) void lib2_entry() { shared_allocate(); }
