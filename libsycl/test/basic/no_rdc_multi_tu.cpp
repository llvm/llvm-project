// REQUIRES: any-device

// Run a program whose kernels come from two translation units. With
// -fno-sycl-rdc each translation unit is finalized into a device binary of its
// own, so both binaries have to be registered and both kernels have to be found
// at run time.
// RUN: %clangxx -fsycl -fno-sycl-rdc -c %s -o %t.nordc.main.o
// RUN: %clangxx -fsycl -fno-sycl-rdc -c %S/Inputs/no_rdc_multi_tu_second.cpp \
// RUN:   -o %t.nordc.second.o
// RUN: %clangxx -fsycl -fno-sycl-rdc %t.nordc.main.o %t.nordc.second.o \
// RUN:   -o %t.nordc.out
// RUN: %t.nordc.out

// Splitting the device code of a translation unit by kernel gives it more than
// one device image, all of which end up in the fat binary of that translation
// unit. The runtime is handed one binary and has to find the kernels across all
// of its images.
// RUN: %clangxx -fsycl -fno-sycl-rdc -fsycl-device-image-split=kernel -c %s \
// RUN:   -o %t.split.main.o
// RUN: %clangxx -fsycl -fno-sycl-rdc -fsycl-device-image-split=kernel -c \
// RUN:   %S/Inputs/no_rdc_multi_tu_second.cpp -o %t.split.second.o
// RUN: %clangxx -fsycl -fno-sycl-rdc %t.split.main.o %t.split.second.o \
// RUN:   -o %t.split.out
// RUN: %t.split.out

// An RDC build of the same sources has to give the same result.
// RUN: %clangxx -fsycl -fsycl-rdc -c %s -o %t.rdc.main.o
// RUN: %clangxx -fsycl -fsycl-rdc -c %S/Inputs/no_rdc_multi_tu_second.cpp \
// RUN:   -o %t.rdc.second.o
// RUN: %clangxx -fsycl -fsycl-rdc %t.rdc.main.o %t.rdc.second.o -o %t.rdc.out
// RUN: %t.rdc.out

#include "Inputs/no_rdc_multi_tu_second.hpp"

#include <cassert>
#include <cstddef>

static void runFirstTuKernel(sycl::queue &Q, int *Data, std::size_t N) {
  Q.parallel_for<class FirstTuKernel>(N,
                                      [=](sycl::item<1> I) { Data[I] += 1; });
}

static void runFirstTuOtherKernel(sycl::queue &Q, int *Data, std::size_t N) {
  Q.parallel_for<class FirstTuOtherKernel>(
      N, [=](sycl::item<1> I) { Data[I] += 2; });
}

int main() {
  constexpr std::size_t N = 16;

  sycl::queue Q;
  int *Data = sycl::malloc_shared<int>(N, Q);
  for (std::size_t I = 0; I < N; ++I)
    Data[I] = static_cast<int>(I);

  runFirstTuKernel(Q, Data, N);
  Q.wait();
  runFirstTuOtherKernel(Q, Data, N);
  Q.wait();
  runSecondTuKernel(Q, Data, N);
  Q.wait();

  for (std::size_t I = 0; I < N; ++I)
    assert(Data[I] == (static_cast<int>(I) + 3) * 3);

  sycl::free(Data, Q);
  return 0;
}
