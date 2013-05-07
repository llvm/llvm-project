//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class Facet> const Facet& use_facet(const locale& loc);

#include <locale>
#include <cassert>

int facet_count = 0;

struct my_facet
    : public std::locale::facet
{
    static std::locale::id id;

    bool im_alive;

    my_facet() : im_alive(true) {++facet_count;}
    ~my_facet() {im_alive = false; --facet_count;}
};

std::locale::id my_facet::id;

int main()
{
    try
    {
        const my_facet& f = std::use_facet<my_facet>(std::locale());
        assert(false);
    }
    catch (std::bad_cast&)
    {
    }
    const my_facet* fp = 0;
    {
        std::locale loc(std::locale(), new my_facet);
        const my_facet& f = std::use_facet<my_facet>(loc);
        assert(f.im_alive);
        fp = &f;
        assert(fp->im_alive);
        assert(facet_count == 1);
    }
    assert(facet_count == 0);
}
