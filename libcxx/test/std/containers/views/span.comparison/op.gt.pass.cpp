// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17 

// <span>

// template<class T, ptrdiff_t X, class U, ptrdiff_t Y>
//   constexpr bool operator>(span<T, X> l, span<U, Y> r);
//   
//
// Effects: Equivalent to: return (r < l);
//

#include <span>
#include <cassert>

#include "test_macros.h"

struct A{};
bool operator==(A, A) {return true;}

constexpr   int iArr1[] = { 0,  1,  2,  1,  2,  5,  6,  7,  8,  9};
            int iArr2[] = { 0,  1,  2,  1,  2,  5,  6,  7,  8,  9};
constexpr float fArr1[]  = {0., 1., 2., 1., 2., 5., 6., 7., 8., 9.};
          float fArr2[]  = {0., 1., 2., 1., 2., 5., 6., 7., 8., 9.};
         

int main () {

    constexpr std::span<const int>     csp0d{};
    constexpr std::span<const int>     csp1d{iArr1, 10};
    constexpr std::span<const int>     csp2d{iArr1 + 3, 2};
    constexpr std::span<const int>     csp3d{iArr1 + 1, 2};
    constexpr std::span<const int>     csp4d{iArr1 + 6, 2};

    constexpr std::span<const int,  0> csp0s{};
    constexpr std::span<const int, 10> csp1s{iArr1, 10};
    constexpr std::span<const int, 2>  csp2s{iArr1 + 3, 2};
    constexpr std::span<const int, 2>  csp3s{iArr1 + 1, 2};
    constexpr std::span<const int, 2>  csp4s{iArr1 + 6, 2};

    static_assert(!(csp0d > csp0d), "");
    static_assert(!(csp0s > csp0s), "");
    static_assert(!(csp0s > csp0d), "");
    static_assert(!(csp0d > csp0s), "");
    
    static_assert(!(csp0d > csp1d), "");
    static_assert(!(csp0s > csp1s), "");
    static_assert(!(csp0s > csp1d), "");
    static_assert(!(csp0d > csp1s), "");
    
    static_assert(!(csp1d > csp1s), "");
    static_assert(!(csp1s > csp1d), "");
    
    static_assert(!(csp2d > csp3d), "");
    static_assert(!(csp2s > csp3s), "");
    static_assert(!(csp2d > csp3s), "");
    static_assert(!(csp2s > csp3d), "");

    static_assert(!(csp2d > csp4d), "");
    static_assert(!(csp2s > csp4s), "");
    static_assert(!(csp2d > csp4s), "");
    static_assert(!(csp2s > csp4d), "");

    static_assert( (csp4d > csp2d), "");
    static_assert( (csp4s > csp2s), "");
    static_assert( (csp4d > csp2s), "");
    static_assert( (csp4s > csp2d), "");

    std::span<int>     sp0d{};
    std::span<int>     sp1d{iArr2, 10};
    std::span<int>     sp2d{iArr2 + 3, 2};
    std::span<int>     sp3d{iArr2 + 1, 2};
    std::span<int>     sp4d{iArr2 + 6, 2};

    std::span<int,  0> sp0s{};
    std::span<int, 10> sp1s{iArr2, 10};
    std::span<int, 2>  sp2s{iArr2 + 3, 2};
    std::span<int, 2>  sp3s{iArr2 + 1, 2};
    std::span<int, 2>  sp4s{iArr2 + 6, 2};

    assert(!(sp0d > sp0d));
    assert(!(sp0s > sp0s));
    assert(!(sp0s > sp0d));
    assert(!(sp0d > sp0s));
    
    assert(!(sp0d > sp1d));
    assert(!(sp0s > sp1s));
    assert(!(sp0s > sp1d));
    assert(!(sp0d > sp1s));
    
    assert(!(sp1d > sp1s));
    assert(!(sp1s > sp1d));
    
    assert(!(sp2d > sp3d));
    assert(!(sp2s > sp3s));
    assert(!(sp2d > sp3s));
    assert(!(sp2s > sp3d));

    assert(!(sp2d > sp4d));
    assert(!(sp2s > sp4s));
    assert(!(sp2d > sp4s));
    assert(!(sp2s > sp4d));

    assert( (sp4d > sp2d));
    assert( (sp4s > sp2s));
    assert( (sp4d > sp2s));
    assert( (sp4s > sp2d));

//  cross type comparisons
    assert(!(csp0d > sp0d));
    assert(!(csp0s > sp0s));
    assert(!(csp0s > sp0d));
    assert(!(csp0d > sp0s));
    
    assert(!(csp0d > sp1d));
    assert(!(csp0s > sp1s));
    assert(!(csp0s > sp1d));
    assert(!(csp0d > sp1s));
    
    assert(!(csp1d > sp1s));
    assert(!(csp1s > sp1d));
    
    assert(!(csp2d > sp3d));
    assert(!(csp2s > sp3s));
    assert(!(csp2d > sp3s));
    assert(!(csp2s > sp3d));

    assert(!(csp2d > sp4d));
    assert(!(csp2s > sp4s));
    assert(!(csp2d > sp4s));
    assert(!(csp2s > sp4d));
    
    assert( (csp4d > sp2d));
    assert( (csp4s > sp2s));
    assert( (csp4d > sp2s));
    assert( (csp4s > sp2d));


//  More cross-type comparisons (int vs float)
    static_assert(!(std::span<const float>{fArr1, 8} > std::span<const   int>{iArr1, 9}), "");
    static_assert(!(std::span<const   int>{iArr1, 8} > std::span<const float>{fArr1, 9}), "");
    assert(!(std::span<float>{fArr2} > std::span<int>{iArr2}));
    assert(!(std::span<int>{iArr2} > std::span<float>{fArr2}));

    static_assert( (std::span<const   int>{iArr1, 9} > std::span<const float>{fArr1, 8}), "");
}