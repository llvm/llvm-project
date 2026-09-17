// RUN: %clangxx -fsycl %s -o %t.out -Wno-error=deprecated-declarations
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>

int main() {
  sycl::nd_range<1> OneDimNdRangeOffset({4}, {2}, {1});
  assert(OneDimNdRangeOffset.get_global_range() == sycl::range<1>(4));
  assert(OneDimNdRangeOffset.get_local_range() == sycl::range<1>(2));
  assert(OneDimNdRangeOffset.get_group_range() == sycl::range<1>(2));
  assert(OneDimNdRangeOffset.get_offset() == sycl::id<1>(1));
  std::cout << "one_dim_nd_range_offset passed " << std::endl;

  sycl::nd_range<2> TwoDimNdRangeOffset({8, 16}, {4, 8}, {1, 1});
  assert(TwoDimNdRangeOffset.get_global_range() == sycl::range<2>(8, 16));
  assert(TwoDimNdRangeOffset.get_local_range() == sycl::range<2>(4, 8));
  assert(TwoDimNdRangeOffset.get_group_range() == sycl::range<2>(2, 2));
  assert(TwoDimNdRangeOffset.get_offset() == sycl::id<2>(1, 1));
  std::cout << "two_dim_nd_range_offset passed " << std::endl;

  sycl::nd_range<3> ThreeDimNdRangeOffset({32, 64, 128}, {16, 32, 64},
                                          {1, 1, 1});
  assert(ThreeDimNdRangeOffset.get_global_range() ==
         sycl::range<3>(32, 64, 128));
  assert(ThreeDimNdRangeOffset.get_local_range() == sycl::range<3>(16, 32, 64));
  assert(ThreeDimNdRangeOffset.get_group_range() == sycl::range<3>(2, 2, 2));
  assert(ThreeDimNdRangeOffset.get_offset() == sycl::id<3>(1, 1, 1));
  std::cout << "three_dim_nd_range_offset passed " << std::endl;

  sycl::nd_range<1> OneDimNdRange({4}, {2});
  assert(OneDimNdRange.get_global_range() == sycl::range<1>(4));
  assert(OneDimNdRange.get_local_range() == sycl::range<1>(2));
  assert(OneDimNdRange.get_group_range() == sycl::range<1>(2));
  assert(OneDimNdRange.get_offset() == sycl::id<1>(0));
  std::cout << "one_dim_nd_range passed " << std::endl;

  sycl::nd_range<2> TwoDimNdRange({8, 16}, {4, 8});
  assert(TwoDimNdRange.get_global_range() == sycl::range<2>(8, 16));
  assert(TwoDimNdRange.get_local_range() == sycl::range<2>(4, 8));
  assert(TwoDimNdRange.get_group_range() == sycl::range<2>(2, 2));
  assert(TwoDimNdRange.get_offset() == sycl::id<2>(0, 0));
  std::cout << "two_dim_nd_range passed " << std::endl;

  sycl::nd_range<3> ThreeDimNdRange({32, 64, 128}, {16, 32, 64});
  assert(ThreeDimNdRange.get_global_range() == sycl::range<3>(32, 64, 128));
  assert(ThreeDimNdRange.get_local_range() == sycl::range<3>(16, 32, 64));
  assert(ThreeDimNdRange.get_group_range() == sycl::range<3>(2, 2, 2));
  assert(ThreeDimNdRange.get_offset() == sycl::id<3>(0, 0, 0));
  std::cout << "three_dim_nd_range passed " << std::endl;
}
