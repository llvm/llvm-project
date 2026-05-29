// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

using namespace sycl;

int main() {
  sycl::range<1> OneDimRange(64);
  sycl::range<2> TwoDimRange(64, 1);
  sycl::range<3> ThreeDimRange(64, 1, 2);
  assert(OneDimRange.size() == 64);
  assert(OneDimRange.get(0) == 64);
  assert(OneDimRange[0] == 64);
  assert(TwoDimRange.size() == 64);
  assert(TwoDimRange.get(0) == 64);
  assert(TwoDimRange[0] == 64);
  assert(TwoDimRange.get(1) == 1);
  assert(TwoDimRange[1] == 1);
  assert(ThreeDimRange.size() == 128);
  assert(ThreeDimRange.get(0) == 64);
  assert(ThreeDimRange[0] == 64);
  assert(ThreeDimRange.get(1) == 1);
  assert(ThreeDimRange[1] == 1);
  assert(ThreeDimRange.get(2) == 2);
  assert(ThreeDimRange[2] == 2);

  sycl::range<3> Default3;
  sycl::range<2> Default2;
  sycl::range<1> Default1;

  assert(Default3[0] == 0 && Default3[1] == 0 && Default3[2] == 0);
  assert(Default2[0] == 0 && Default2[1] == 0);
  assert(Default1[0] == 0);

  const range<3> A(8, 9, 10);
  const range<3> B(2, 3, 5);

  assert((A + B) == range<3>(10, 12, 15));
  assert((A - B) == range<3>(6, 6, 5));
  assert((A * B) == range<3>(16, 27, 50));
  assert((A / B) == range<3>(4, 3, 2));
  assert((A % B) == range<3>(0, 0, 0));
  assert((A << B) == range<3>(32, 72, 320));
  assert((A >> B) == range<3>(2, 1, 0));
  assert((A & B) == range<3>(0, 1, 0));
  assert((A | B) == range<3>(10, 11, 15));
  assert((A ^ B) == range<3>(10, 10, 15));
  assert((A && B) == range<3>(1, 1, 1));
  assert((A || B) == range<3>(1, 1, 1));
  assert((A < B) == range<3>(0, 0, 0));
  assert((A > B) == range<3>(1, 1, 1));
  assert((A <= B) == range<3>(0, 0, 0));
  assert((A >= B) == range<3>(1, 1, 1));

  assert((A + 1) == range<3>(9, 10, 11));
  assert((1 + A) == range<3>(9, 10, 11));
  assert((A - 1) == range<3>(7, 8, 9));
  assert((20 - B) == range<3>(18, 17, 15));
  assert((A * 2) == range<3>(16, 18, 20));
  assert((2 * A) == range<3>(16, 18, 20));
  assert((A / 2) == range<3>(4, 4, 5));
  assert((20 / B) == range<3>(10, 6, 4));
  assert((A % 4) == range<3>(0, 1, 2));
  assert((33 % B) == range<3>(1, 0, 3));
  assert((A << 1) == range<3>(16, 18, 20));
  assert((1 << B) == range<3>(4, 8, 32));
  assert((A >> 1) == range<3>(4, 4, 5));
  assert((256 >> B) == range<3>(64, 32, 8));
  assert((A & 6) == range<3>(0, 0, 2));
  assert((15 & B) == range<3>(2, 3, 5));
  assert((A | 1) == range<3>(9, 9, 11));
  assert((1 | B) == range<3>(3, 3, 5));
  assert((A ^ 3) == range<3>(11, 10, 9));
  assert((3 ^ B) == range<3>(1, 0, 6));
  assert((A && 0) == range<3>(0, 0, 0));
  assert((0 || B) == range<3>(1, 1, 1));
  assert((A < 9) == range<3>(1, 0, 0));
  assert((9 > A) == range<3>(1, 0, 0));
  assert((A <= 9) == range<3>(1, 1, 0));
  assert((9 >= A) == range<3>(1, 1, 0));

  range<3> E = A;
  E += B;
  assert(E == range<3>(10, 12, 15));
  E -= B;
  assert(E == A);

  E *= B;
  assert(E == range<3>(16, 27, 50));
  E /= B;
  assert(E == A);
  E %= B;
  assert(E == range<3>(0, 0, 0));

  range<3> F(8, 9, 10);
  F += 2;
  assert(F == range<3>(10, 11, 12));
  F -= 2;
  assert(F == range<3>(8, 9, 10));
  F *= 2;
  assert(F == range<3>(16, 18, 20));
  F /= 2;
  assert(F == range<3>(8, 9, 10));
  F %= 6;
  assert(F == range<3>(2, 3, 4));
  F <<= 1;
  assert(F == range<3>(4, 6, 8));
  F >>= 1;
  assert(F == range<3>(2, 3, 4));
  F &= 6;
  assert(F == range<3>(2, 2, 4));
  F |= 1;
  assert(F == range<3>(3, 3, 5));
  F ^= 2;
  assert(F == range<3>(1, 1, 7));

  range<3> U(8, 9, 10);
  assert(+U == range<3>(8, 9, 10));
  assert(-U == range<3>(static_cast<std::size_t>(-8),
                        static_cast<std::size_t>(-9),
                        static_cast<std::size_t>(-10)));
  assert(++U == range<3>(9, 10, 11));
  assert(U++ == range<3>(9, 10, 11));
  assert(U == range<3>(10, 11, 12));
  assert(--U == range<3>(9, 10, 11));
  assert(U-- == range<3>(9, 10, 11));
  assert(U == range<3>(8, 9, 10));

  return 0;
}
