//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class discrete_distribution

// discrete_distribution(initializer_list<double> wl);

#include <random>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        typedef std::discrete_distribution<> D;
        D d = {1., 2.};
        std::vector<double> p = d.probabilities();
        assert(p.size() == 2);
        assert(p[0] == 1);
        assert(p[1] == 2);
    }
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
