// RUN: %check_clang_tidy %s modernize-use-std-tie %t

#include <tuple>

struct A {
    int n;
    double s; 
    float w;
    float v;
    float d() const noexcept { return w / v; }
};

// CHECK-MESSAGES: :[[@LINE+1]]:6: warning: use std::tie to implement lexicographical comparison [modernize-use-std-tie]
bool operator<(const A& lhs, const A& rhs) noexcept
{
    if (lhs.n != rhs.n) {
        return lhs.n < rhs.n;
    } else if (lhs.s != rhs.s) {
        return lhs.s < rhs.s;
    }
    return lhs.d() < rhs.d();
}
// CHECK-FIXES: bool operator<(const A& lhs, const A& rhs) noexcept
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT:     const auto lhs_d = lhs.d();
// CHECK-FIXES-NEXT:     const auto rhs_d = rhs.d();
// CHECK-FIXES-NEXT:     return std::tie(lhs.n, lhs.s, lhs_d) < std::tie(rhs.n, rhs.s, rhs_d);
// CHECK-FIXES-NEXT: }

// CHECK-MESSAGES: :[[@LINE+1]]:6: warning: use std::tie to implement lexicographical comparison [modernize-use-std-tie]
bool operator>(const A& lhs, const A& rhs) noexcept
{
    if (lhs.n != rhs.n) {
        return lhs.n > rhs.n;
    } else if (lhs.s != rhs.s) {
        return lhs.s > rhs.s;
    }
    return lhs.d() > rhs.d();
}
// CHECK-FIXES: bool operator>(const A& lhs, const A& rhs) noexcept
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT:     const auto lhs_d = lhs.d();
// CHECK-FIXES-NEXT:     const auto rhs_d = rhs.d();
// CHECK-FIXES-NEXT:     return std::tie(lhs.n, lhs.s, lhs_d) > std::tie(rhs.n, rhs.s, rhs_d);
// CHECK-FIXES-NEXT: }