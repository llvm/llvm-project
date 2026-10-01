// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <type_traits>

int main() {
  // TODO: uncomment property once it is implemented. now all sycl::queue
  // objects are in-order due to liboffload limitation. Test is intended to
  // check in-order execution.
  sycl::queue Q{/*sycl::property::queue::in_order()*/};
  auto Dev = Q.get_device();
  auto Ctx = Q.get_context();
  constexpr int N = 8;

  auto A = static_cast<int *>(sycl::malloc_shared(N * sizeof(int), Dev, Ctx));

  for (int I = 0; I < N; ++I)
    A[I] = 1;

  Q.parallel_for<class IntRange>(N, [=](auto I) {
    static_assert(std::is_same<decltype(I), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
    A[I]++;
  });

  Q.parallel_for<class InitRange>({N}, [=](auto I) {
    static_assert(std::is_same<decltype(I), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
    A[I]++;
  });

  Q.parallel_for<class InitRange2D>({4, 2}, [=](auto I) {
    static_assert(std::is_same<decltype(I), sycl::item<2>>::value,
                  "lambda arg type is unexpected");
    A[I.get_linear_id()]++;
  });

  Q.parallel_for<class InitRange3D>({2, 2, 2}, [=](auto I) {
    static_assert(std::is_same<decltype(I), sycl::item<3>>::value,
                  "lambda arg type is unexpected");
    A[I.get_linear_id()]++;
  });

  sycl::nd_range<1> NDR(sycl::range<1>{N}, sycl::range<1>{2});
  Q.parallel_for<class NdRange1D>(NDR, [=](auto NdItem) {
    static_assert(std::is_same<decltype(NdItem), sycl::nd_item<1>>::value,
                  "lambda arg type is unexpected");
    A[NdItem.get_global_id()]++;
  });

  sycl::nd_range<2> NDR2D(sycl::range<2>{4, 2}, sycl::range<2>{2, 1});
  Q.parallel_for<class NdRange2D>(NDR2D, [=](auto NdItem) {
    static_assert(std::is_same<decltype(NdItem), sycl::nd_item<2>>::value,
                  "lambda arg type is unexpected");
    A[NdItem.get_global_linear_id()]++;
  });

  sycl::nd_range<3> NDR3D(sycl::range<3>{2, 2, 2}, sycl::range<3>{1, 2, 2});
  Q.parallel_for<class NdRange3D>(NDR3D, [=](auto NdItem) {
    static_assert(std::is_same<decltype(NdItem), sycl::nd_item<3>>::value,
                  "lambda arg type is unexpected");
    A[NdItem.get_global_linear_id()]++;
  });

  Q.wait();

  bool Fail{};
  for (int I = 0; I < N; I++)
    Fail |= !(A[I] == 8);
  sycl::free(A, Ctx);
  return Fail;
}
