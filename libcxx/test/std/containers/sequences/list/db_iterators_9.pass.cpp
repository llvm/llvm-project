//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <list>

// Operations on "NULL" iterators

#if _LIBCPP_DEBUG >= 1

#define _LIBCPP_ASSERT(x, m) do { if (!x) throw 1; } while(0)

#include <list>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

struct S { int val; };

int main()
{
    {
    unsigned lib_asserts;

    typedef S T;
    typedef std::list<T> C;
    C::iterator i{};
    C::const_iterator ci{};

    lib_asserts = 0;
    try { ++i; }  catch (int) { ++lib_asserts; }
    try { i++; }  catch (int) { ++lib_asserts; }
    try { ++ci; } catch (int) { ++lib_asserts; }
    try { ci++; } catch (int) { ++lib_asserts; }
    assert(lib_asserts == 4);

    lib_asserts = 0;
    try { --i; }  catch (int) { ++lib_asserts; }
    try { i--; }  catch (int) { ++lib_asserts; }
    try { --ci; } catch (int) { ++lib_asserts; }
    try { ci--; } catch (int) { ++lib_asserts; }
    assert(lib_asserts == 4);

    lib_asserts = 0;
    try { *i; }             catch (int) { ++lib_asserts; }
    try { *ci; }            catch (int) { ++lib_asserts; }
    try { (void)  i->val; } catch (int) { ++lib_asserts; }
    try { (void) ci->val; } catch (int) { ++lib_asserts; }
    assert(lib_asserts == 4);
    }
}

#else

int main()
{
}

#endif
