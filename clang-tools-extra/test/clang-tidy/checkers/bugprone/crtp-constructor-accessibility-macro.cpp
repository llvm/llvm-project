// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-crtp-constructor-accessibility %t -- -- -fno-delayed-template-parsing

// Keep 'CONSTEXPR' the first expansion in the file. The crash needs a file
// entry right before it, and a split '>>' or another macro above would hide it.
#define CONSTEXPR constexpr

namespace ctor_begins_with_macro {
template <typename T>
class CRTP {
public:
    CONSTEXPR CRTP() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CONSTEXPR CRTP() {}{{[[:space:]]*}}public:
    // CHECK-FIXES: friend T;
};

class A : CRTP<A> {};
} // namespace ctor_begins_with_macro

#define MEMBERS(Name) int I; Name() {}

namespace ctor_inside_macro {
template <typename T>
class CRTP {
public:
    MEMBERS(CRTP)
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: MEMBERS(CRTP)
    // CHECK-FIXES-NOT: friend T;
};

// CHECK-FIXES: class A : CRTP<A> {};
class A : CRTP<A> {};
} // namespace ctor_inside_macro
