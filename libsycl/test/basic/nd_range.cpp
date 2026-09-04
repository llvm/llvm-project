// RUN: %clangxx -fsycl %s -o %t.out -Wno-error=deprecated-declarations
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>

int main() {
  sycl::nd_range<1> one_dim_nd_range_offset({4}, {2}, {1});
  assert(one_dim_nd_range_offset.get_global_range() == sycl::range<1>(4));
  assert(one_dim_nd_range_offset.get_local_range() == sycl::range<1>(2));
  assert(one_dim_nd_range_offset.get_group_range() == sycl::range<1>(2));
  assert(one_dim_nd_range_offset.get_offset() == sycl::id<1>(1));
  std::cout << "one_dim_nd_range_offset passed " << std::endl;

  sycl::nd_range<2> two_dim_nd_range_offset({8, 16}, {4, 8}, {1, 1});
  assert(two_dim_nd_range_offset.get_global_range() == sycl::range<2>(8, 16));
  assert(two_dim_nd_range_offset.get_local_range() == sycl::range<2>(4, 8));
  assert(two_dim_nd_range_offset.get_group_range() == sycl::range<2>(2, 2));
  assert(two_dim_nd_range_offset.get_offset() == sycl::id<2>(1, 1));
  std::cout << "two_dim_nd_range_offset passed " << std::endl;

  sycl::nd_range<3> three_dim_nd_range_offset({32, 64, 128}, {16, 32, 64},
                                              {1, 1, 1});
  assert(three_dim_nd_range_offset.get_global_range() ==
         sycl::range<3>(32, 64, 128));
  assert(three_dim_nd_range_offset.get_local_range() ==
         sycl::range<3>(16, 32, 64));
  assert(three_dim_nd_range_offset.get_group_range() ==
         sycl::range<3>(2, 2, 2));
  assert(three_dim_nd_range_offset.get_offset() == sycl::id<3>(1, 1, 1));
  std::cout << "three_dim_nd_range_offset passed " << std::endl;

  sycl::nd_range<1> one_dim_nd_range({4}, {2});
  assert(one_dim_nd_range.get_global_range() == sycl::range<1>(4));
  assert(one_dim_nd_range.get_local_range() == sycl::range<1>(2));
  assert(one_dim_nd_range.get_group_range() == sycl::range<1>(2));
  assert(one_dim_nd_range.get_offset() == sycl::id<1>(0));
  std::cout << "one_dim_nd_range passed " << std::endl;

  sycl::nd_range<2> two_dim_nd_range({8, 16}, {4, 8});
  assert(two_dim_nd_range.get_global_range() == sycl::range<2>(8, 16));
  assert(two_dim_nd_range.get_local_range() == sycl::range<2>(4, 8));
  assert(two_dim_nd_range.get_group_range() == sycl::range<2>(2, 2));
  assert(two_dim_nd_range.get_offset() == sycl::id<2>(0, 0));
  std::cout << "two_dim_nd_range passed " << std::endl;

  sycl::nd_range<3> three_dim_nd_range({32, 64, 128}, {16, 32, 64});
  assert(three_dim_nd_range.get_global_range() == sycl::range<3>(32, 64, 128));
  assert(three_dim_nd_range.get_local_range() == sycl::range<3>(16, 32, 64));
  assert(three_dim_nd_range.get_group_range() == sycl::range<3>(2, 2, 2));
  assert(three_dim_nd_range.get_offset() == sycl::id<3>(0, 0, 0));
  std::cout << "three_dim_nd_range passed " << std::endl;
}
