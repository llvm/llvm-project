// RUN: %check_clang_tidy %s bugprone-return-const-ref-from-parameter %t

using T = int;
using TConst = int const;
using TConstRef = int const&;

namespace invalid {

int const &f1(int const &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: returning a constant reference parameter

int const &f2(T const &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: returning a constant reference parameter

int const &f3(TConstRef a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: returning a constant reference parameter

int const &f4(TConst &a) { return a; }
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: returning a constant reference parameter

} // namespace invalid

namespace valid {

int const &f1(int &a) { return a; }

int const &f2(int &&a) { return a; }

int f1(int const &a) { return a; }

} // namespace valid
