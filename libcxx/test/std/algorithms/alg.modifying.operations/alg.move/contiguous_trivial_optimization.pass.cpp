//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Older compilers don't support std::is_constant_evaluated
// UNSUPPORTED: clang-4, clang-5, clang-6, clang-7, clang-8
// UNSUPPORTED: apple-clang-9, apple-clang-10
// UNSUPPORTED: c++03

// <algorithm>

// We optimize std::copy(_backward) and std::move(_backward) into memmove
// when the iterator is trivial and contiguous and the type in question
// is also trivially (copyable, movable). This test verifies that the
// optimization never eliminates an actually non-trivial copy or move.

#include <algorithm>
#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct TMBNTC {
    int *p;
    constexpr TMBNTC(int& copies) : p(&copies) {}
    constexpr TMBNTC(const TMBNTC&) = default;
    TEST_CONSTEXPR_CXX14 TMBNTC& operator=(TMBNTC&&) = default;
    TEST_CONSTEXPR_CXX14 TMBNTC& operator=(const TMBNTC&) { ++*p; return *this; }
};

TEST_CONSTEXPR_CXX20 bool
test_trivial_moveassign_but_no_trivial_copyassign()
{
    int copies = 0;
    TMBNTC ia[] = { copies, copies, copies, copies };
    TMBNTC ib[] = { copies, copies, copies, copies };
    std::copy(ia, ia+4, ib);
    assert(copies == 4);
    copies = 0;
    std::copy_backward(ia, ia+4, ib+4);
    assert(copies == 4);

    copies = 0;
    std::copy(std::make_move_iterator(ia), std::make_move_iterator(ia+4), ib);
    assert(copies == 0);
    std::copy_backward(std::make_move_iterator(ia), std::make_move_iterator(ia+4), ib+4);
    assert(copies == 0);

    std::move(ia, ia+4, ib);
    assert(copies == 0);
    std::move_backward(ia, ia+4, ib+4);
    assert(copies == 0);

    return true;
}

struct TCBNTM {
    int *p;
    constexpr TCBNTM(int& moves) : p(&moves) {}
    constexpr TCBNTM(const TCBNTM&) = default;
    TEST_CONSTEXPR_CXX14 TCBNTM& operator=(TCBNTM&&) { ++*p; return *this; }
    TEST_CONSTEXPR_CXX14 TCBNTM& operator=(const TCBNTM&) = default;
};

TEST_CONSTEXPR_CXX20 bool
test_trivial_copyassign_but_no_trivial_moveassign()
{
    int moves = 0;
    TCBNTM ia[] = { moves, moves, moves, moves };
    TCBNTM ib[] = { moves, moves, moves, moves };
    std::move(ia, ia+4, ib);
    assert(moves == 4);
    moves = 0;
    std::move_backward(ia, ia+4, ib+4);
    assert(moves == 4);

    moves = 0;
    std::copy(std::make_move_iterator(ia), std::make_move_iterator(ia+4), ib);
    assert(moves == 4);
    moves = 0;
    std::copy_backward(std::make_move_iterator(ia), std::make_move_iterator(ia+4), ib+4);
    assert(moves == 4);

    moves = 0;
    std::copy(ia, ia+4, ib);
    assert(moves == 0);
    std::copy_backward(ia, ia+4, ib+4);
    assert(moves == 0);

    return true;
}

int main(int, char**)
{
    test_trivial_moveassign_but_no_trivial_copyassign();
    test_trivial_copyassign_but_no_trivial_moveassign();

#if TEST_STD_VER > 17
    static_assert(test_trivial_moveassign_but_no_trivial_copyassign());
    static_assert(test_trivial_copyassign_but_no_trivial_moveassign());
#endif

    return 0;
}
