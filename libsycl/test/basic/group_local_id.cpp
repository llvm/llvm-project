// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

template <int Dims> class group_local_id_kernel;

template <int Dims>
bool runGroupLocalIdCase(sycl::queue &Q, sycl::nd_range<Dims> ExecRange,
                         size_t Count) {
  int *Out = sycl::malloc_shared<int>(Count, Q);
  for (size_t I = 0; I < Count; ++I)
    Out[I] = -1;

  Q.parallel_for<group_local_id_kernel<Dims>>(
      ExecRange, [=](sycl::nd_item<Dims> Item) {
        Out[Item.get_global_linear_id()] =
            (Item.get_local_id() == Item.get_group().get_local_id());
      });

  Q.wait();

  bool Match = true;
  for (size_t I = 0; I < Count; ++I)
    Match &= (Out[I] == 1);

  sycl::free(Out, Q);
  return !Match;
}

int main() {
  sycl::queue Q;
  constexpr size_t N = 8;

  bool Failure = runGroupLocalIdCase<1>(
      Q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{N}}, N);
  Failure |= runGroupLocalIdCase<2>(
      Q, sycl::nd_range<2>{sycl::range<2>{N, N}, sycl::range<2>{N, N}}, N * N);
  Failure |= runGroupLocalIdCase<3>(
      Q, sycl::nd_range<3>{sycl::range<3>{N, N, N}, sycl::range<3>{N, N, N}},
      N * N * N);

  return Failure;
}
