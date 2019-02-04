//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class numpunct;

// string grouping() const;

#include <locale>
#include <cassert>

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        typedef char C;
        const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
        assert(np.grouping() == std::string());
    }
    {
        typedef wchar_t C;
        const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
        assert(np.grouping() == std::string());
    }

  return 0;
}
