// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <sycl/sycl.hpp>

using sycl::detail::Builder;

int main() {
  // Default construction initializes all dimensions to zero.
  sycl::id<1> Zero1;
  sycl::id<2> Zero2;
  sycl::id<3> Zero3;
  assert(Zero1 == sycl::id<1>(0));
  assert(Zero2 == sycl::id<2>(0, 0));
  assert(Zero3 == sycl::id<3>(0, 0, 0));

  // Dimensional constructors and element access.
  sycl::id<1> I1(64);
  sycl::id<2> I2(128, 256);
  sycl::id<3> I3(64, 1, 2);
  assert(I1.get(0) == 64 && I1[0] == 64);
  assert(I2.get(0) == 128 && I2.get(1) == 256);
  assert(I3.get(0) == 64 && I3.get(1) == 1 && I3.get(2) == 2);

  // Construction from range.
  sycl::id<1> FromRange1(sycl::range<1>(2));
  sycl::id<2> FromRange2(sycl::range<2>(4, 8));
  sycl::id<3> FromRange3(sycl::range<3>(16, 32, 64));
  assert(FromRange1 == sycl::id<1>(2));
  assert(FromRange2 == sycl::id<2>(4, 8));
  assert(FromRange3 == sycl::id<3>(16, 32, 64));

  // Construction from item preserves computed id coordinates.
  sycl::item<1, true> Item1 = Builder::createItem<1, true>({4}, {2}, {1});
  sycl::item<2, true> Item2 =
      Builder::createItem<2, true>({8, 16}, {4, 8}, {1, 1});
  sycl::item<3, true> Item3 =
      Builder::createItem<3, true>({32, 64, 128}, {16, 32, 64}, {1, 1, 1});
  assert(sycl::id<1>(Item1) == sycl::id<1>(2));
  assert(sycl::id<2>(Item2) == sycl::id<2>(4, 8));
  assert(sycl::id<3>(Item3) == sycl::id<3>(16, 32, 64));

  // Equality and inequality semantics.
  assert(sycl::id<1>(10) == sycl::id<1>(10));
  assert(sycl::id<2>(10, 15) != sycl::id<2>(10, 12));
  assert(sycl::id<3>(1, 2, 3) == sycl::id<3>(1, 2, 3));
  assert(sycl::id<1>(10) == 10);
  assert(10 == sycl::id<1>(10));
  assert(sycl::id<1>(10) != 19);

  // 1D id implicit conversion to scalar.
  sycl::id<1> OneDimCast(16);
  std::size_t S = OneDimCast;
  int I = OneDimCast;
  assert(S == 16 && I == 16);

  // Representative unary/increment behavior.
  sycl::id<2> Unary(64, 1);
  assert(+Unary == sycl::id<2>(64, 1));
  assert(-Unary == sycl::id<2>(static_cast<std::size_t>(-64),
                               static_cast<std::size_t>(-1)));
  assert(++Unary == sycl::id<2>(65, 2));
  assert(Unary++ == sycl::id<2>(65, 2));
  assert(Unary == sycl::id<2>(66, 3));
  assert(--Unary == sycl::id<2>(65, 2));
  assert(Unary-- == sycl::id<2>(65, 2));
  assert(Unary == sycl::id<2>(64, 1));

  return 0;
}
