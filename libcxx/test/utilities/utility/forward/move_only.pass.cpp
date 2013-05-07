//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test move

#include <utility>
#include <cassert>

class move_only
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    move_only(const move_only&);
    move_only& operator=(const move_only&);
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    move_only(move_only&);
    move_only& operator=(move_only&);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

public:

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    move_only(move_only&&) {}
    move_only& operator=(move_only&&) {return *this;}
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    operator std::__rv<move_only> () {return std::__rv<move_only>(*this);}
    move_only(std::__rv<move_only>) {}
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

    move_only() {}
};

move_only source() {return move_only();}
const move_only csource() {return move_only();}

void test(move_only) {}

int main()
{
    move_only mo;

    test(std::move(mo));
    test(source());
}
