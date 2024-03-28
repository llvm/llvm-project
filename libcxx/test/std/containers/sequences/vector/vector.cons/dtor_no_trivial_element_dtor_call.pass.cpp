//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>
#include <cassert>

struct int_alloc
{
    typedef int value_type;

    template< class U >
    struct rebind
    {
        typedef int_alloc other;
    };

    int* allocate(std::size_t n)
    {
        return new int[n];
    }
    void deallocate(int* p, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i) {
            assert(p[i] != 0);
        }
        delete p;
    }
};

struct with_dtor
{
    with_dtor() = default;
    with_dtor(const with_dtor &) = default;

    int x = 42;

    ~with_dtor() {
        x = 0;
    }
};

struct with_dtor_alloc
{
    typedef with_dtor value_type;

    template< class U >
    struct rebind
    {
        typedef with_dtor_alloc other;
    };

    with_dtor* allocate(std::size_t n)
    {
        return new with_dtor[n];
    }
    void deallocate(with_dtor* p, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i) {
            assert(p[i].x == 0);
        }
        delete[] p;
    }
};

void tests()
{
    {
        std::vector<with_dtor, with_dtor_alloc> v(5, with_dtor());
    }
    {
        std::vector<int, int_alloc> v(5, 42);
    }
}

int main(int, char**)
{
    tests();
    return 0;
}

