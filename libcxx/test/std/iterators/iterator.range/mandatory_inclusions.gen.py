# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# In addition to being available via inclusion of the <iterator> header,
# the function templates in [iterator.range] are available when any of the following
# headers are included: <array>, <deque>, <flat_map>, <flat_set>, <forward_list>,
# <list>, <map>, <regex>, <set>, <span>, <string>, <string_view>, <unordered_map>,
# <unordered_set>, <vector>.

# UNSUPPORTED: c++03

# RUN: %{python} %s %{libcxx-dir}/utils
# END.

import sys

sys.path.append(sys.argv[1])
from libcxx.header_information import (
    lit_header_restrictions,
    lit_header_undeprecations,
    Header,
)

headers = list(
    map(
        Header,
        [
            "array",
            "deque",
            "flat_map",
            "flat_set",
            "forward_list",
            "list",
            "map",
            "regex",
            "set",
            "span",
            "string",
            "string_view",
            "unordered_map",
            "unordered_set",
            "vector",
        ],
    )
)

for header in headers:
    print(
        f"""\
//--- {header}.pass.cpp
{lit_header_restrictions.get(header, '')}
{lit_header_undeprecations.get(header, '')}

#include <{header}>
#include <cassert>
#include <initializer_list>

#include "test_macros.h"

struct Container {{
  int a[3] = {{1, 2, 3}};

  int* begin() {{ return &a[0]; }}
  const int* begin() const {{ return &a[0]; }}
  int* rbegin() {{ return &a[2]; }}
  const int* rbegin() const {{ return &a[2]; }}
  int* end() {{ return &a[3]; }}
  const int* end() const {{ return &a[3]; }}
  int* rend() {{ return (&a[0]) - 1; }}
  const int* rend() const {{ return (&a[0]) - 1; }}
  std::size_t size() const {{ return 3; }}
  bool empty() const {{ return false; }}
  int* data() {{return &a[0]; }}
  const int* data() const {{ return &a[0]; }}
}};

int main(int, char**)  {{
  {{
    Container c;
    const auto& cc = c;
    assert(std::begin(c) == c.begin());
    assert(std::begin(cc) == cc.begin());
    assert(std::end(c) == c.end());
    assert(std::end(cc) == cc.end());
#if TEST_STD_VER >= 14
    assert(std::cbegin(c) == cc.begin());
    assert(std::cbegin(cc) == cc.begin());
    assert(std::cend(c) == cc.end());
    assert(std::cend(cc) == cc.end());
    assert(std::rbegin(c) == c.rbegin());
    assert(std::rbegin(cc) == cc.rbegin());
    assert(std::rend(c) == cc.rend());
    assert(std::rend(cc) == cc.rend());
    assert(std::crbegin(c) == cc.rbegin());
    assert(std::crbegin(cc) == cc.rbegin());
    assert(std::crend(c) == cc.rend());
    assert(std::crend(cc) == cc.rend());
#endif
#if TEST_STD_VER >= 17
    assert(std::data(c) == c.data());
    assert(std::data(cc) == cc.data());
    assert(std::size(cc) == cc.size());
    assert(std::empty(cc) == cc.empty());
#endif
#if TEST_STD_VER >= 20
    assert(std::ssize(cc) == 3);
#endif
  }}
  {{
    int a[]        = {{1, 2, 3}};
    const auto& ca = a;
    assert(std::begin(a) == &a[0]);
    assert(std::begin(ca) == &ca[0]);
    assert(std::end(a) == &a[3]);
    assert(std::end(ca) == &ca[3]);
#if TEST_STD_VER >= 14
    assert(std::cbegin(a) == &a[0]);
    assert(std::cbegin(ca) == &ca[0]);
    assert(std::cend(a) == &a[3]);
    assert(std::cend(ca) == &ca[3]);
    assert(std::rbegin(a) == std::reverse_iterator<int*>(std::end(a)));
    assert(std::rbegin(ca) == std::reverse_iterator<const int*>(std::end(ca)));
    assert(std::rend(a) == std::reverse_iterator<int*>(std::begin(a)));
    assert(std::rend(ca) == std::reverse_iterator<const int*>(std::begin(ca)));
    assert(std::crbegin(a) == std::reverse_iterator<const int*>(std::end(a)));
    assert(std::crbegin(ca) == std::reverse_iterator<const int*>(std::end(ca)));
    assert(std::crend(a) == std::reverse_iterator<const int*>(std::begin(a)));
    assert(std::crend(ca) == std::reverse_iterator<const int*>(std::begin(ca)));
#endif
#if TEST_STD_VER >= 17
    assert(std::size(ca) == 3);
    assert(!std::empty(ca));
    assert(std::data(a) == &a[0]);
    assert(std::data(ca) == &ca[0]);
#endif
#if TEST_STD_VER >= 20
    assert(std::ssize(ca) == 3);
#endif
  }}
  {{
    auto il         = {{1, 2, 3}};
    const auto& cil = il;
    assert(std::begin(il) == il.begin());
    assert(std::begin(cil) == cil.begin());
    assert(std::end(il) == il.end());
    assert(std::end(cil) == cil.end());
#if TEST_STD_VER >= 14
    assert(std::cbegin(il) == cil.begin());
    assert(std::cbegin(cil) == cil.begin());
    assert(std::cend(il) == cil.end());
    assert(std::cend(cil) == cil.end());
    assert(std::rbegin(il) == std::reverse_iterator<const int*>(std::end(il)));
    assert(std::rbegin(cil) == std::reverse_iterator<const int*>(std::end(il)));
    assert(std::rend(il) == std::reverse_iterator<const int*>(std::begin(il)));
    assert(std::rend(cil) == std::reverse_iterator<const int*>(std::begin(il)));
    assert(std::crbegin(il) == std::reverse_iterator<const int*>(std::end(il)));
    assert(std::crbegin(cil) == std::reverse_iterator<const int*>(std::end(il)));
    assert(std::crend(il) == std::reverse_iterator<const int*>(std::begin(il)));
    assert(std::crend(cil) == std::reverse_iterator<const int*>(std::begin(il)));
#endif
#if TEST_STD_VER >= 17
    assert(std::size(cil) == 3);
    assert(!std::empty(cil));
    assert(std::data(il) == &*std::begin(il));
    assert(std::data(cil) == &*std::begin(il));
#endif
#if TEST_STD_VER >= 20
    assert(std::ssize(cil) == 3);
#endif
  }}

  return 0;
}}

"""
    )
