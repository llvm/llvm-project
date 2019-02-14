//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_get<charT, InputIterator>

// explicit num_get(size_t refs = 0);

#include <locale>
#include <cassert>

typedef std::num_get<char, char*> F;

class my_facet
    : public F
{
public:
    static int count;

    explicit my_facet(std::size_t refs = 0)
        : F(refs) {++count;}

    ~my_facet() {--count;}
};

int my_facet::count = 0;

int main(int, char**)
{
    {
        std::locale l(std::locale::classic(), new my_facet);
        assert(my_facet::count == 1);
    }
    assert(my_facet::count == 0);
    {
        my_facet f(1);
        assert(my_facet::count == 1);
        {
            std::locale l(std::locale::classic(), &f);
            assert(my_facet::count == 1);
        }
        assert(my_facet::count == 1);
    }
    assert(my_facet::count == 0);

  return 0;
}
