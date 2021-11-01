//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// bool operator==(const vector& lhs, const vector& rhs);
// bool operator!=(const vector& lhs, const vector& rhs);
// bool operator< (const vector& lhs, const vector& rhs);
// bool operator<=(const vector& lhs, const vector& rhs);
// bool operator> (const vector& lhs, const vector& rhs);
// bool operator>=(const vector& lhs, const vector& rhs);

#include <vector>
#include <cassert>

#include "test_comparisons.h"

int main(int, char**) {
    {
        const std::vector<int> c1, c2;
        assert(testComparisons6(c1, c2, true, false));
    }
    {
        const std::vector<int> c1(1, 1), c2(1, 2);
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        const std::vector<int> c1, c2(1, 2);
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        int items1[3] = {1, 2, 1};
        int items2[3] = {1, 2, 2};
        const std::vector<int> c1(items1, items1 + 3);
        const std::vector<int> c2(items2, items2 + 3);
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        int items1[3] = {3, 2, 3};
        int items2[3] = {3, 1, 3};
        const std::vector<int> c1(items1, items1 + 3);
        const std::vector<int> c2(items2, items2 + 3);

        assert(testComparisons6(c1, c2, false, false));
    }
    {
        int items1[2] = {1, 2};
        int items2[3] = {1, 2, 0};
        const std::vector<int> c1(items1, items1 + 2);
        const std::vector<int> c2(items2, items2 + 3);
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        int items1[3] = {1, 2, 0};
        const std::vector<int> c1(items1, items1 + 3);
        const std::vector<int> c2(1, 3);
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        const std::vector<LessAndEqComp> c1, c2;
        assert(testComparisons6(c1, c2, true, false));
    }
    {
        const std::vector<LessAndEqComp> c1(1, LessAndEqComp(1));
        const std::vector<LessAndEqComp> c2(1, LessAndEqComp(1));
        assert(testComparisons6(c1, c2, true, false));
    }
    {
        const std::vector<LessAndEqComp> c1(1, LessAndEqComp(1));
        const std::vector<LessAndEqComp> c2(1, LessAndEqComp(2));
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        const std::vector<LessAndEqComp> c1;
        const std::vector<LessAndEqComp> c2(1, LessAndEqComp(2));
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        LessAndEqComp items1[3] = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(2)};
        LessAndEqComp items2[3] = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(1)};
        const std::vector<LessAndEqComp> c1(items1, items1 + 3);
        const std::vector<LessAndEqComp> c2(items2, items2 + 3);
        assert(testComparisons6(c1, c2, false, false));
    }
    {
        LessAndEqComp items1[3] = {LessAndEqComp(3), LessAndEqComp(3), LessAndEqComp(3)};
        LessAndEqComp items2[3] = {LessAndEqComp(3), LessAndEqComp(2), LessAndEqComp(3)};
        const std::vector<LessAndEqComp> c1(items1, items1 + 3);
        const std::vector<LessAndEqComp> c2(items2, items2 + 3);
        assert(testComparisons6(c1, c2, false, false));
    }
    {
        LessAndEqComp items1[2] = {LessAndEqComp(1), LessAndEqComp(2)};
        LessAndEqComp items2[3] = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(0)};
        const std::vector<LessAndEqComp> c1(items1, items1 + 2);
        const std::vector<LessAndEqComp> c2(items2, items2 + 3);
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        LessAndEqComp items1[3] = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(0)};
        const std::vector<LessAndEqComp> c1(items1, items1 + 3);
        const std::vector<LessAndEqComp> c2(1, LessAndEqComp(3));
        assert(testComparisons6(c1, c2, false, true));
    }
    {
        assert((std::vector<int>() == std::vector<int>()));
        assert(!(std::vector<int>() != std::vector<int>()));
        assert(!(std::vector<int>() < std::vector<int>()));
        assert((std::vector<int>() <= std::vector<int>()));
        assert(!(std::vector<int>() > std::vector<int>()));
        assert((std::vector<int>() >= std::vector<int>()));
    }

    return 0;
}
