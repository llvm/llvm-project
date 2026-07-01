// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// Unified test for sycl::range and sycl::id covering all operators defined in
// theirs base class, plus class-specific behaviour for each type.

#include <cassert>
#include <sycl/sycl.hpp>
#include <type_traits>

using sycl::detail::Builder;

// Helper to create values for any dimension from three components.
// Only uses the required number of values based on dimension.
template <template <int> class T, int Dim>
T<Dim> makeValue(std::size_t a, std::size_t b = 0, std::size_t c = 0) {
  if constexpr (Dim == 1)
    return T<1>(a);
  else if constexpr (Dim == 2)
    return T<2>(a, b);
  else
    return T<3>(a, b, c);
}

// Tests binary and compound operators for a specific dimension.
template <template <int> class T, int Dim> void testBinaryOpsForDim() {
  const T<Dim> P = makeValue<T, Dim>(8, 9, 10);
  const T<Dim> Q = makeValue<T, Dim>(2, 3, 5);

  // Binary operators: object op object.
  assert((P + Q) == (makeValue<T, Dim>(10, 12, 15)));
  assert((P - Q) == (makeValue<T, Dim>(6, 6, 5)));
  assert((P * Q) == (makeValue<T, Dim>(16, 27, 50)));
  assert((P / Q) == (makeValue<T, Dim>(4, 3, 2)));
  assert((P % Q) == (makeValue<T, Dim>(0, 0, 0)));
  assert((P << Q) == (makeValue<T, Dim>(32, 72, 320)));
  assert((P >> Q) == (makeValue<T, Dim>(2, 1, 0)));
  assert((P & Q) == (makeValue<T, Dim>(0, 1, 0)));
  assert((P | Q) == (makeValue<T, Dim>(10, 11, 15)));
  assert((P ^ Q) == (makeValue<T, Dim>(10, 10, 15)));
  assert((P && Q) == (makeValue<T, Dim>(1, 1, 1)));
  assert((P || Q) == (makeValue<T, Dim>(1, 1, 1)));
  assert((P < Q) == (makeValue<T, Dim>(0, 0, 0)));
  assert((P > Q) == (makeValue<T, Dim>(1, 1, 1)));
  assert((P <= Q) == (makeValue<T, Dim>(0, 0, 0)));
  assert((P >= Q) == (makeValue<T, Dim>(1, 1, 1)));

  // Binary operators: object op scalar.
  assert((P + 1) == (makeValue<T, Dim>(9, 10, 11)));
  assert((P - 1) == (makeValue<T, Dim>(7, 8, 9)));
  assert((P * 2) == (makeValue<T, Dim>(16, 18, 20)));
  assert((P / 2) == (makeValue<T, Dim>(4, 4, 5)));
  assert((P % 4) == (makeValue<T, Dim>(0, 1, 2)));
  assert((P << 1) == (makeValue<T, Dim>(16, 18, 20)));
  assert((P >> 1) == (makeValue<T, Dim>(4, 4, 5)));
  assert((P & 6) == (makeValue<T, Dim>(0, 0, 2)));
  assert((P | 1) == (makeValue<T, Dim>(9, 9, 11)));
  assert((P ^ 3) == (makeValue<T, Dim>(11, 10, 9)));
  assert((P && 0) == (makeValue<T, Dim>(0, 0, 0)));
  assert((P || 0) == (makeValue<T, Dim>(1, 1, 1)));
  assert((P < 9) == (makeValue<T, Dim>(1, 0, 0)));
  assert((P > 9) == (makeValue<T, Dim>(0, 0, 1)));
  assert((P <= 9) == (makeValue<T, Dim>(1, 1, 0)));
  assert((P >= 9) == (makeValue<T, Dim>(0, 1, 1)));

  // Binary operators: scalar op object.
  assert((1 + P) == (makeValue<T, Dim>(9, 10, 11)));
  assert((20 - Q) == (makeValue<T, Dim>(18, 17, 15)));
  assert((2 * P) == (makeValue<T, Dim>(16, 18, 20)));
  assert((20 / Q) == (makeValue<T, Dim>(10, 6, 4)));
  assert((33 % Q) == (makeValue<T, Dim>(1, 0, 3)));
  assert((1 << Q) == (makeValue<T, Dim>(4, 8, 32)));
  assert((256 >> Q) == (makeValue<T, Dim>(64, 32, 8)));
  assert((15 & Q) == (makeValue<T, Dim>(2, 3, 5)));
  assert((1 | Q) == (makeValue<T, Dim>(3, 3, 5)));
  assert((3 ^ Q) == (makeValue<T, Dim>(1, 0, 6)));
  assert((1 && Q) == (makeValue<T, Dim>(1, 1, 1)));
  assert((0 || Q) == (makeValue<T, Dim>(1, 1, 1)));
  assert((9 < P) == (makeValue<T, Dim>(0, 0, 1)));
  assert((9 > P) == (makeValue<T, Dim>(1, 0, 0)));
  assert((9 <= P) == (makeValue<T, Dim>(0, 1, 1)));
  assert((9 >= P) == (makeValue<T, Dim>(1, 1, 0)));

  // Compound assignment: object op= object.
  T<Dim> E = makeValue<T, Dim>(8, 9, 10);
  E += Q;
  assert(E == (makeValue<T, Dim>(10, 12, 15)));
  E -= Q;
  assert(E == (makeValue<T, Dim>(8, 9, 10)));
  E *= Q;
  assert(E == (makeValue<T, Dim>(16, 27, 50)));
  E /= Q;
  assert(E == (makeValue<T, Dim>(8, 9, 10)));
  E %= Q;
  assert(E == (makeValue<T, Dim>(0, 0, 0)));

  // Compound assignment: object op= scalar.
  T<Dim> F = makeValue<T, Dim>(8, 9, 10);
  F += 2;
  assert(F == (makeValue<T, Dim>(10, 11, 12)));
  F -= 2;
  assert(F == (makeValue<T, Dim>(8, 9, 10)));
  F *= 2;
  assert(F == (makeValue<T, Dim>(16, 18, 20)));
  F /= 2;
  assert(F == (makeValue<T, Dim>(8, 9, 10)));
  F %= 6;
  assert(F == (makeValue<T, Dim>(2, 3, 4)));
  F <<= 1;
  assert(F == (makeValue<T, Dim>(4, 6, 8)));
  F >>= 1;
  assert(F == (makeValue<T, Dim>(2, 3, 4)));
  F &= 6;
  assert(F == (makeValue<T, Dim>(2, 2, 4)));
  F |= 1;
  assert(F == (makeValue<T, Dim>(3, 3, 5)));
  F ^= 2;
  assert(F == (makeValue<T, Dim>(1, 1, 7)));
}

template <template <int> class T> void testCommon() {
  // Default construction zero-initialises every dimension.
  T<1> Def1;
  T<2> Def2;
  T<3> Def3;
  assert(Def1[0] == 0);
  assert(Def2[0] == 0 && Def2[1] == 0);
  assert(Def3[0] == 0 && Def3[1] == 0 && Def3[2] == 0);

  // Dimensional constructors and element access via get() and const [].
  T<1> A1(5);
  T<2> A2(3, 7);
  T<3> A3(8, 9, 10);
  assert(A1.get(0) == 5 && A1[0] == 5);
  assert(A2.get(0) == 3 && A2.get(1) == 7);
  assert(A3.get(0) == 8 && A3.get(1) == 9 && A3.get(2) == 10);

  // dimensions static constexpr.
  static_assert(T<1>::dimensions == 1);
  static_assert(T<2>::dimensions == 2);
  static_assert(T<3>::dimensions == 3);

  // Non-const operator[] write.
  T<2> Mutable(1, 2);
  Mutable[0] = 10;
  Mutable[1] = 20;
  assert(Mutable[0] == 10 && Mutable[1] == 20);

  // Copy constructor and copy assignment.
  T<3> CopyCtor(A3);
  assert(CopyCtor == A3);
  T<3> CopyAssign(1, 1, 1);
  CopyAssign = A3;
  assert(CopyAssign == A3);

  // Move constructor and move assignment.
  T<3> MoveSrc1(A3);
  T<3> MoveCtor(std::move(MoveSrc1));
  assert(MoveCtor == A3);
  T<3> MoveSrc2(A3);
  T<3> MoveAssign(1, 1, 1);
  MoveAssign = std::move(MoveSrc2);
  assert(MoveAssign == A3);

  // Equality and inequality between same-type objects.
  assert((makeValue<T, 3>(8, 9, 10)) == (makeValue<T, 3>(8, 9, 10)));
  assert((makeValue<T, 3>(8, 9, 10)) != (makeValue<T, 3>(8, 9, 11)));

  // Test binary and compound operators for all dimensions.
  testBinaryOpsForDim<T, 1>();
  testBinaryOpsForDim<T, 2>();
  testBinaryOpsForDim<T, 3>();

  // Unary + and -.
  T<3> U(8, 9, 10);
  assert((+U) == (makeValue<T, 3>(8, 9, 10)));
  assert((-U) == (makeValue<T, 3>(static_cast<std::size_t>(-8),
                                  static_cast<std::size_t>(-9),
                                  static_cast<std::size_t>(-10))));

  // Pre- and post-increment / decrement.
  T<3> Inc(8, 9, 10);
  assert((++Inc) == (makeValue<T, 3>(9, 10, 11)));
  assert((Inc++) == (makeValue<T, 3>(9, 10, 11)));
  assert(Inc == (makeValue<T, 3>(10, 11, 12)));
  assert((--Inc) == (makeValue<T, 3>(9, 10, 11)));
  assert((Inc--) == (makeValue<T, 3>(9, 10, 11)));
  assert(Inc == (makeValue<T, 3>(8, 9, 10)));
}

void testRange() {
  testCommon<sycl::range>();

  assert(sycl::range<1>(7).size() == 7);
  assert(sycl::range<2>(4, 5).size() == 20);
  assert(sycl::range<3>(2, 3, 4).size() == 24);
  assert(sycl::range<3>(0, 5, 3).size() == 0);

#ifdef __cpp_deduction_guides
  auto R1 = sycl::range(7);
  auto R2 = sycl::range(4, 5);
  auto R3 = sycl::range(2, 3, 4);
  static_assert(std::is_same_v<decltype(R1), sycl::range<1>>);
  static_assert(std::is_same_v<decltype(R2), sycl::range<2>>);
  static_assert(std::is_same_v<decltype(R3), sycl::range<3>>);
  assert(R1 == sycl::range<1>(7));
  assert(R2 == sycl::range<2>(4, 5));
  assert(R3 == sycl::range<3>(2, 3, 4));
#endif
}

void testId() {
  testCommon<sycl::id>();

  // Construction from range.
  assert(sycl::id<1>(sycl::range<1>(2)) == sycl::id<1>(2));
  assert(sycl::id<2>(sycl::range<2>(4, 8)) == sycl::id<2>(4, 8));
  assert(sycl::id<3>(sycl::range<3>(16, 32, 64)) == sycl::id<3>(16, 32, 64));

  // Construction from item (WithOffset=true) preserves the computed id
  // coordinates.
  sycl::item<1, true> Item1 = Builder::createItem<1, true>({4}, {2}, {1});
  sycl::item<2, true> Item2 =
      Builder::createItem<2, true>({8, 16}, {4, 8}, {1, 1});
  sycl::item<3, true> Item3 =
      Builder::createItem<3, true>({32, 64, 128}, {16, 32, 64}, {1, 1, 1});
  assert(sycl::id<1>(Item1) == sycl::id<1>(2));
  assert(sycl::id<2>(Item2) == sycl::id<2>(4, 8));
  assert(sycl::id<3>(Item3) == sycl::id<3>(16, 32, 64));

  // Construction from item (WithOffset=false).
  sycl::item<1, false> Item1NoOffset = Builder::createItem<1, false>({4}, {2});
  sycl::item<2, false> Item2NoOffset =
      Builder::createItem<2, false>({8, 16}, {4, 8});
  sycl::item<3, false> Item3NoOffset =
      Builder::createItem<3, false>({32, 64, 128}, {16, 32, 64});
  assert(sycl::id<1>(Item1NoOffset) == sycl::id<1>(2));
  assert(sycl::id<2>(Item2NoOffset) == sycl::id<2>(4, 8));
  assert(sycl::id<3>(Item3NoOffset) == sycl::id<3>(16, 32, 64));

  // Copy-initialization from item must also select id(item) constructor.
  sycl::id<1> IdFromItem1NoOffset = Item1NoOffset;
  sycl::id<2> IdFromItem2NoOffset = Item2NoOffset;
  sycl::id<3> IdFromItem3NoOffset = Item3NoOffset;
  assert(IdFromItem1NoOffset == sycl::id<1>(2));
  assert(IdFromItem2NoOffset == sycl::id<2>(4, 8));
  assert(IdFromItem3NoOffset == sycl::id<3>(16, 32, 64));

#ifdef __cpp_deduction_guides
  auto I1 = sycl::id(2);
  auto I2 = sycl::id(4, 8);
  auto I3 = sycl::id(16, 32, 64);
  static_assert(std::is_same_v<decltype(I1), sycl::id<1>>);
  static_assert(std::is_same_v<decltype(I2), sycl::id<2>>);
  static_assert(std::is_same_v<decltype(I3), sycl::id<3>>);
  assert(I1 == sycl::id<1>(2));
  assert(I2 == sycl::id<2>(4, 8));
  assert(I3 == sycl::id<3>(16, 32, 64));
#endif

  // Implicit conversion to size_t (id<1> only).
  sycl::id<1> OneDim(16);
  std::size_t S = OneDim;
  int I = OneDim;
  assert(S == 16 && I == 16);

  // Scalar equality/inequality operators (id<1> only).
  const sycl::id<1> ConstId(10);
  assert(ConstId == 10);
  assert(10 == ConstId);
  assert(ConstId != 19);
  assert(19 != ConstId);
  assert(ConstId == 10u);
  assert(10u == ConstId);
  assert(ConstId != static_cast<short>(9));
  assert(static_cast<short>(9) != ConstId);
}

int main() {
  testRange();
  testId();
  return 0;
}
