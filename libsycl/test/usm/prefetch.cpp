// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cstddef>

using namespace sycl;

constexpr std::size_t Count = 1024;
constexpr std::size_t NumBytes = Count * sizeof(int);

int main() {
  queue Q;

  int *SharedData = malloc_shared<int>(Count, Q);
  assert(SharedData != nullptr);

  for (std::size_t I = 0; I < Count; ++I)
    SharedData[I] = static_cast<int>(I);

  event E1 = Q.prefetch(SharedData, NumBytes);

  event E2 = Q.prefetch(SharedData, NumBytes / 2, E1);

  event E3 = Q.prefetch(nullptr, 0, E2);
  E3.wait();

  Q.parallel_for(range<1>{Count}, [=](id<1> Idx) {
     SharedData[Idx] += 1;
   }).wait();

  for (std::size_t I = 0; I < Count; ++I)
    assert(SharedData[I] == static_cast<int>(I + 1));

  free(SharedData, Q);

  return 0;
}
