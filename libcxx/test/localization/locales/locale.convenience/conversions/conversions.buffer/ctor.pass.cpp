//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// wbuffer_convert<Codecvt, Elem, Tr>

// wbuffer_convert(streambuf *bytebuf = 0, Codecvt *pcvt = new Codecvt,
//                 state_type state = state_type());

#include <locale>
#include <codecvt>
#include <sstream>
#include <cassert>
#include <new>

int new_called = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    ++new_called;
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    --new_called;
    std::free(p);
}

int main()
{
    typedef std::wbuffer_convert<std::codecvt_utf8<wchar_t> > B;
    {
        B b;
        assert(b.rdbuf() == nullptr);
        assert(new_called != 0);
    }
    assert(new_called == 0);
    {
        std::stringstream s;
        B b(s.rdbuf());
        assert(b.rdbuf() == s.rdbuf());
        assert(new_called != 0);
    }
    assert(new_called == 0);
    {
        std::stringstream s;
        B b(s.rdbuf(), new std::codecvt_utf8<wchar_t>);
        assert(b.rdbuf() == s.rdbuf());
        assert(new_called != 0);
    }
    assert(new_called == 0);
    {
        std::stringstream s;
        B b(s.rdbuf(), new std::codecvt_utf8<wchar_t>, std::mbstate_t());
        assert(b.rdbuf() == s.rdbuf());
        assert(new_called != 0);
    }
    assert(new_called == 0);
}
