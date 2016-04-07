//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// static void swap(reference x, reference y) noexcept;

#include <vector>
#include <cassert>

int main()
{

    bool a[] = {false, true, false, true};
    bool* an = a + sizeof(a)/sizeof(a[0]);

	std::vector<bool> v(a, an);
	std::vector<bool>::reference r1 = v[0];
	std::vector<bool>::reference r2 = v[3];

#if __has_feature(cxx_noexcept)
    static_assert((noexcept(v.swap(r1,r2))), "");
#endif

	assert(!r1);
	assert( r2);
	v.swap(r1, r2);
	assert( r1);
	assert(!r2);
}