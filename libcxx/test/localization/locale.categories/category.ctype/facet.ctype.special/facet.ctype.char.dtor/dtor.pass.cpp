//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>

// ~ctype();

#include <locale>
#include <cassert>
#include <new>

unsigned delete_called = 0;

void operator delete[](void* p) throw()
{
    operator delete(p);
    ++delete_called;
}

int main()
{
    {
        delete_called = 0;
        std::locale l(std::locale::classic(), new std::ctype<char>);
        assert(delete_called == 0);
    }
    assert(delete_called == 0);
    {
        std::ctype<char>::mask table[256];
        delete_called = 0;
        std::locale l(std::locale::classic(), new std::ctype<char>(table));
        assert(delete_called == 0);
    }
    assert(delete_called == 0);
    {
        delete_called = 0;
        std::locale l(std::locale::classic(),
            new std::ctype<char>(new std::ctype<char>::mask[256], true));
        assert(delete_called == 0);
    }
    assert(delete_called == 1);
}
