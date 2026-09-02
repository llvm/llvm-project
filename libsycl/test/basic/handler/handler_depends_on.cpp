// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>

void testMemcpyDependency() {
  constexpr size_t N = 4;
  sycl::queue Queue;

  int *Src = sycl::malloc_shared<int>(N, Queue);
  int *Dst = sycl::malloc_shared<int>(N, Queue);
  std::iota(Src, Src + N, 1);
  std::fill(Dst, Dst + N, 0);

  auto MemCpyEvent = Queue.submit(
      [&](sycl::handler &CGH) { CGH.memcpy(Dst, Src, N * sizeof(int)); });

  auto *DstElementCopy = sycl::malloc_shared<int>(1, Queue);
  *DstElementCopy = 0;

  auto KernelEvent = Queue.submit([&](sycl::handler &CGH) {
    CGH.depends_on(MemCpyEvent);
    CGH.single_task<class DependsOnMemcpyKernel>(
        [=]() { *DstElementCopy = Dst[0]; });
  });

  KernelEvent.wait();

  assert(Dst[0] == 1 && Dst[1] == 2 && Dst[2] == 3 && Dst[3] == 4);
  assert(*DstElementCopy == 1);

  sycl::free(Src, Queue);
  sycl::free(Dst, Queue);
  sycl::free(DstElementCopy, Queue);
}

void testNDRangeDependency() {
  constexpr size_t N = 16;
  sycl::queue Queue;

  int *Data = sycl::malloc_shared<int>(N, Queue);
  int *Token = sycl::malloc_shared<int>(1, Queue);

  std::fill(Data, Data + N, 0);
  *Token = 0;

  auto InitEvent =
      Queue.single_task<class DependsOnNDRangeInit>([=]() { *Token = 9; });

  auto KernelEvent = Queue.submit([&](sycl::handler &CGH) {
    CGH.depends_on(InitEvent);
    CGH.parallel_for<class DependsOnNDRangeKernel>(
        sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{4}},
        [=](sycl::nd_item<1> Item) {
          const size_t I = Item.get_global_id(0);
          Data[I] = static_cast<int>(I) + *Token;
        });
  });

  KernelEvent.wait();

  for (size_t I = 0; I < N; ++I)
    assert(Data[I] == static_cast<int>(I) + 9);

  sycl::free(Data, Queue);
  sycl::free(Token, Queue);
}

int main() {
  testMemcpyDependency();
  testNDRangeDependency();
  return 0;
}
