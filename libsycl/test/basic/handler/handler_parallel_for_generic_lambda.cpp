// RUN: %clangxx -fsycl -fsyntax-only %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

#include <type_traits>

template <typename KernelName, typename ExpectedType, typename Range>
void test_parallel_for(Range r) {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<KernelName>(r, [=](auto item) {
      static_assert(std::is_same<decltype(item), ExpectedType>::value,
                    "Argument type is unexpected");
    });
  });
}

int main() {
  test_parallel_for<class Item1Name, sycl::item<1>>(sycl::range<1>{1});
  test_parallel_for<class Item2Name, sycl::item<2>>(sycl::range<2>{1, 1});
  test_parallel_for<class Item3Name, sycl::item<3>>(sycl::range<3>{1, 1, 1});
  test_parallel_for<class NDItem1Name, sycl::nd_item<1>>(
      sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}});
  test_parallel_for<class NDItem2Name, sycl::nd_item<2>>(
      sycl::nd_range<2>{sycl::range<2>{2, 2}, sycl::range<2>{1, 1}});
  test_parallel_for<class NDItem3Name, sycl::nd_item<3>>(
      sycl::nd_range<3>{sycl::range<3>{2, 2, 2}, sycl::range<3>{1, 1, 1}});

  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class GenericInitList1>(sycl::range{1}, [=](auto &) {});
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class GenericInitList2>(sycl::range{1, 1}, [=](auto &) {});
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class GenericInitList3>(sycl::range{1, 1, 1},
                                             [=](auto &) {});
  });

  return 0;
}
