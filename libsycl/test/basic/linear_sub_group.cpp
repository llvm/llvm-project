// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

int main() {
  sycl::queue Q;

  constexpr uint32_t Outer = 2;
  constexpr uint32_t Inner = 8;
  constexpr uint32_t Size = Outer * Inner;

  int *Output = sycl::malloc_shared<int>(Size, Q);
  for (uint32_t I = 0; I < Size; ++I)
    Output[I] = -1;

  Q.parallel_for<class linear_sub_group>(
      sycl::nd_range<2>(sycl::range<2>(Outer, Inner),
                        sycl::range<2>(Outer, Inner)),
      [=](sycl::nd_item<2> It) {
        sycl::sub_group SG = It.get_sub_group();
        Output[It.get_global_linear_id()] =
            SG.get_group_linear_id() * SG.get_local_linear_range() +
            SG.get_local_linear_id();
      });

  Q.wait();

  for (uint32_t I = 0; I < Size; ++I)
    assert(Output[I] == static_cast<int>(I));

  sycl::free(Output, Q);
  return 0;
}
