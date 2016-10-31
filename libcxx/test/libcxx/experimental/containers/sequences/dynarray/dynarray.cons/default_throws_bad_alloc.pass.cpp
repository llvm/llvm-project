//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// dynarray.cons

// explicit dynarray(size_type c);

// UNSUPPORTED: c++98, c++03, c++11

// The sanitizers replace new/delete with versions that do not throw bad_alloc.
// UNSUPPORTED: sanitizer-new-delete, ubsan


#include <experimental/dynarray>
#include <limits>
#include <new>
#include <cassert>


using std::experimental::dynarray;

int main() {
    try { dynarray<int>((std::numeric_limits<size_t>::max() / sizeof(int)) - 1); }
    catch (std::bad_alloc &) { return 0; }
    catch (...) { assert(false); }
    assert(false);
}
