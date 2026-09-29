// REQUIRES: any-device

// Run a program whose kernels come from two translation units. With
// -fno-gpu-rdc each translation unit is finalized into a device binary of its
// own, so both binaries have to be registered and both kernels have to be found
// at run time.
// The second translation unit is this same source compiled with -DSECOND_TU.
// RUN: %clangxx -fsycl -fno-gpu-rdc -c %s -o %t.nordc.main.o
// RUN: %clangxx -fsycl -fno-gpu-rdc -DSECOND_TU -c %s -o %t.nordc.second.o
// RUN: %clangxx %t.nordc.main.o %t.nordc.second.o -L%sycl_libs_dir -lLLVMSYCL \
// RUN:   -o %t.nordc.out
// RUN: %t.nordc.out

// An RDC build of the same sources has to give the same result.
// RUN: %clangxx -fsycl -fgpu-rdc -c %s -o %t.rdc.main.o
// RUN: %clangxx -fsycl -fgpu-rdc -DSECOND_TU -c %s -o %t.rdc.second.o
// RUN: %clangxx -fsycl %t.rdc.main.o %t.rdc.second.o -o %t.rdc.out
// RUN: %t.rdc.out

#include <cassert>
#include <cstddef>

#include <sycl/sycl.hpp>

#ifdef SECOND_TU

void runSecondTuKernel(sycl::queue &Q, int *Data, std::size_t N) {
  Q.parallel_for<class SecondTuKernel>(N,
                                       [=](sycl::item<1> I) { Data[I] *= 3; });
}

#else

void runSecondTuKernel(sycl::queue &Q, int *Data, std::size_t N);

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

#endif
