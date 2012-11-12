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

// param_type(initializer_list<double> wl);

#include <random>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        typedef std::discrete_distribution<> D;
        typedef D::param_type P;
        P pa = {1};
        std::vector<double> p = pa.probabilities();
        assert(p.size() == 1);
        assert(p[0] == 1);
    }
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
