//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class CharT, class Traits>
//   constexpr OutIter
//   copy(CharT* first, CharT* last, ostreambuf_iterator<CharT, Traits> result);

// UNSUPPORTED: no-localization

#include <algorithm>
#include <cassert>
#include <iterator>
#include <sstream>
#include <type_traits>

#include "stream_types.h"
#include "test_macros.h"

template <class CCharT>
void test() {
  using CharT = typename std::remove_cv<CCharT>::type;
  {
    std::basic_ostringstream<CharT> oss;
    CCharT buff[] = {'B', 'a', 'n', 'a', 'n', 'e'};
    std::copy(std::begin(buff), std::end(buff), std::ostreambuf_iterator<CharT>(oss));
    assert(oss.str() == std::basic_string_view<CharT>(buff, 6));
  }
  {
    failing_streambuf<CharT> fsb(4);
    std::basic_ostream<CharT> oss(&fsb);
    CCharT buff[] = {'B', 'a', 'n', 'a', 'n', 'e'};
    auto res      = std::copy(std::begin(buff), std::end(buff), std::ostreambuf_iterator<CharT>(oss));
    assert(res.failed());
    assert(fsb.str() == std::basic_string_view<CharT>(buff, 4));
  }
}

int main(int, char**) {
  test<char>();
  test<const char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<const wchar_t>();
#endif
  return 0;
}
