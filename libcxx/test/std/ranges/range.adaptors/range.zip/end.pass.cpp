//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end() requires(!(simple-view<Views> && ...))
// constexpr auto end() const requires(range<const Views>&&...)

#include <ranges>
#include <tuple>

#include "types.h"

// ID | simple | common | bidi | random | sized | #views |     v.end()    | as_const(v)
//    |        |        |      | access |       |        |                |   .end()
// ---|--------|--------|------|--------|-------|--------|----------------|---------------
// 1  |   Y    |   Y    |  Y   |    Y   |   Y   |   1    | iterator<true> | iterator<true>
// 2  |   Y    |   Y    |  Y   |    Y   |   Y   |   >1   | iterator<true> | iterator<true>
// 3  |   Y    |   N    |  Y   |    Y   |   N   |   1    | sentinel<true> | sentinel<true>
// 4  |   Y    |   N    |  Y   |    Y   |   N   |   >1   | sentinel<true> | sentinel<true>
// 5  |   Y    |   Y    |  Y   |    N   |   Y   |   1    | iterator<true> | iterator<true>
// 6  |   Y    |   Y    |  Y   |    N   |   Y   |   >1   | sentinel<true> | sentinel<true>
// 7  |   Y    |   Y    |  Y   |    N   |   N   |   1    | iterator<true> | iterator<true>
// 8  |   Y    |   Y    |  Y   |    N   |   N   |   >1   | sentinel<true> | sentinel<true>
// 9  |   Y    |   Y    |  N   |    N   |   Y   |   1    | iterator<true> | iterator<true>
// 10 |   Y    |   Y    |  N   |    N   |   Y   |   >1   | iterator<true> | iterator<true>
// 11 |   Y    |   Y    |  N   |    N   |   N   |   1    | iterator<true> | iterator<true>
// 12 |   Y    |   Y    |  N   |    N   |   N   |   >1   | iterator<true> | iterator<true>
// 13 |   Y    |   N    |  Y   |    Y   |   Y   |   1    | iterator<true> | iterator<true>
// 14 |   Y    |   N    |  Y   |    Y   |   Y   |   >1   | iterator<true> | iterator<true>
// 15 |   Y    |   N    |  Y   |    N   |   Y   |   1    | sentinel<true> | sentinel<true>
// 16 |   Y    |   N    |  Y   |    N   |   Y   |   >1   | sentinel<true> | sentinel<true>
// 17 |   Y    |   N    |  Y   |    N   |   N   |   1    | sentinel<true> | sentinel<true>
// 18 |   Y    |   N    |  Y   |    N   |   N   |   >1   | sentinel<true> | sentinel<true>
// 19 |   Y    |   N    |  N   |    N   |   Y   |   1    | sentinel<true> | sentinel<true>
// 20 |   Y    |   N    |  N   |    N   |   Y   |   >1   | sentinel<true> | sentinel<true>
// 21 |   Y    |   N    |  N   |    N   |   N   |   1    | sentinel<true> | sentinel<true>
// 22 |   Y    |   N    |  N   |    N   |   N   |   >1   | sentinel<true> | sentinel<true>
// 23 |   N    |   Y    |  Y   |    Y   |   Y   |   1    | iterator<false>| iterator<true>
// 24 |   N    |   Y    |  Y   |    Y   |   Y   |   >1   | iterator<false>| iterator<true>
// 25 |   N    |   N    |  Y   |    Y   |   N   |   1    | sentinel<false>| sentinel<true>
// 26 |   N    |   N    |  Y   |    Y   |   N   |   >1   | sentinel<false>| sentinel<true>
// 27 |   N    |   Y    |  Y   |    N   |   Y   |   1    | iterator<false>| iterator<true>
// 28 |   N    |   Y    |  Y   |    N   |   Y   |   >1   | sentinel<false>| sentinel<true>
// 29 |   N    |   Y    |  Y   |    N   |   N   |   1    | iterator<false>| iterator<true>
// 30 |   N    |   Y    |  Y   |    N   |   N   |   >1   | sentinel<false>| sentinel<true>
// 31 |   N    |   Y    |  N   |    N   |   Y   |   1    | iterator<false>| iterator<true>
// 32 |   N    |   Y    |  N   |    N   |   Y   |   >1   | iterator<false>| iterator<true>
// 33 |   N    |   Y    |  N   |    N   |   N   |   1    | iterator<false>| iterator<true>
// 34 |   N    |   Y    |  N   |    N   |   N   |   >1   | iterator<false>| iterator<true>
// 35 |   N    |   N    |  Y   |    Y   |   Y   |   1    | iterator<false>| iterator<true>
// 36 |   N    |   N    |  Y   |    Y   |   Y   |   >1   | iterator<false>| iterator<true>
// 37 |   N    |   N    |  Y   |    N   |   Y   |   1    | sentinel<false>| sentinel<true>
// 38 |   N    |   N    |  Y   |    N   |   Y   |   >1   | sentinel<false>| sentinel<true>
// 39 |   N    |   N    |  Y   |    N   |   N   |   1    | sentinel<false>| sentinel<true>
// 40 |   N    |   N    |  Y   |    N   |   N   |   >1   | sentinel<false>| sentinel<true>
// 41 |   N    |   N    |  N   |    N   |   Y   |   1    | sentinel<false>| sentinel<true>
// 42 |   N    |   N    |  N   |    N   |   Y   |   >1   | sentinel<false>| sentinel<true>
// 43 |   N    |   N    |  N   |    N   |   N   |   1    | sentinel<false>| sentinel<true>
// 44 |   N    |   N    |  N   |    N   |   N   |   >1   | sentinel<false>| sentinel<true>

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[1] = {1};
  int buffer3[3] = {1, 2, 3};
  {
    // test ID 1
    std::ranges::zip_view v{SimpleCommonRandomAccessSized(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 5 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 2
    std::ranges::zip_view v{SimpleCommonRandomAccessSized(buffer1), SimpleCommonRandomAccessSized(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 1 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 3
    std::ranges::zip_view v{NonSizedRandomAccessView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 5 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 4
    std::ranges::zip_view v{NonSizedRandomAccessView(buffer1), NonSizedRandomAccessView(buffer3)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 3 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 5
    std::ranges::zip_view v{SizedBidiCommon(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 6
    std::ranges::zip_view v{SizedBidiCommon(buffer1), SizedBidiCommon(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 7
    std::ranges::zip_view v{BidiCommonView(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 8
    std::ranges::zip_view v{BidiCommonView(buffer1), BidiCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 9
    std::ranges::zip_view v{ForwardSizedView(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 10
    std::ranges::zip_view v{ForwardSizedView(buffer1), ForwardSizedView(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 11
    std::ranges::zip_view v{InputCommonView(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::ranges::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 12
    std::ranges::zip_view v{InputCommonView(buffer1), InputCommonView(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 13
    std::ranges::zip_view v{SimpleNonCommonRandomAccessSized(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 5 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 14
    std::ranges::zip_view v{SimpleNonCommonRandomAccessSized(buffer1), SimpleNonCommonRandomAccessSized(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 1 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 15
    std::ranges::zip_view v{SizedBidiNonCommonView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 16
    std::ranges::zip_view v{SizedBidiNonCommonView(buffer1), SizedBidiNonCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 17
    std::ranges::zip_view v{BidiNonCommonView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 18
    std::ranges::zip_view v{BidiNonCommonView(buffer1), BidiNonCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 19
    std::ranges::zip_view v{ForwardSizedNonCommon(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 20
    std::ranges::zip_view v{ForwardSizedNonCommon(buffer1), ForwardSizedNonCommon(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 21
    std::ranges::zip_view v{InputNonCommonView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::ranges::next(v.begin(), 5) == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 22
    std::ranges::zip_view v{InputNonCommonView(buffer1), InputNonCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 23
    std::ranges::zip_view v{NonSimpleCommonRandomAccessSized(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 5 == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 24
    std::ranges::zip_view v{NonSimpleCommonRandomAccessSized(buffer1), NonSimpleCommonRandomAccessSized(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 1 == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 25
    std::ranges::zip_view v{NonSimpleNonSizedRandomAccessView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 5 == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 26
    std::ranges::zip_view v{NonSimpleNonSizedRandomAccessView(buffer1), NonSimpleNonSizedRandomAccessView(buffer3)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 3 == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 27
    std::ranges::zip_view v{NonSimpleSizedBidiCommon(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 28
    std::ranges::zip_view v{NonSimpleSizedBidiCommon(buffer1), NonSimpleSizedBidiCommon(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 29
    std::ranges::zip_view v{NonSimpleBidiCommonView(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 30
    std::ranges::zip_view v{NonSimpleBidiCommonView(buffer1), NonSimpleBidiCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 31
    std::ranges::zip_view v{NonSimpleForwardSizedView(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 32
    std::ranges::zip_view v{NonSimpleForwardSizedView(buffer1), NonSimpleForwardSizedView(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 33
    std::ranges::zip_view v{NonSimpleInputCommonView(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(std::ranges::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 34
    std::ranges::zip_view v{NonSimpleInputCommonView(buffer1), NonSimpleInputCommonView(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 35
    std::ranges::zip_view v{NonSimpleNonCommonRandomAccessSized(buffer1)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 5 == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 36
    std::ranges::zip_view v{NonSimpleNonCommonRandomAccessSized(buffer1), NonSimpleNonCommonRandomAccessSized(buffer2)};
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 1 == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 37
    std::ranges::zip_view v{NonSimpleSizedBidiNonCommonView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 38
    std::ranges::zip_view v{NonSimpleSizedBidiNonCommonView(buffer1), NonSimpleSizedBidiNonCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 39
    std::ranges::zip_view v{NonSimpleBidiNonCommonView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 40
    std::ranges::zip_view v{NonSimpleBidiNonCommonView(buffer1), NonSimpleBidiNonCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 41
    std::ranges::zip_view v{NonSimpleForwardSizedNonCommon(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 42
    std::ranges::zip_view v{NonSimpleForwardSizedNonCommon(buffer1), NonSimpleForwardSizedNonCommon(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 43
    std::ranges::zip_view v{NonSimpleInputNonCommonView(buffer1)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(std::ranges::next(v.begin(), 5) == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // test ID 44
    std::ranges::zip_view v{NonSimpleInputNonCommonView(buffer1), NonSimpleInputNonCommonView(buffer2)};
    static_assert(!std::ranges::common_range<decltype(v)>);
    assert(++v.begin() == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }
  {
    // end should go to the minimum length when zip is common and random_access sized
    std::ranges::zip_view v(std::views::iota(0, 4), std::views::iota(0, 8));
    auto it = --(v.end());
    auto [x, y] = *it;
    assert(x == 3);
    assert(y == 3); // y should not go to the end "7"
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
