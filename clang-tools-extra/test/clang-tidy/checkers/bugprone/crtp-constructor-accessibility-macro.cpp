// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-crtp-constructor-accessibility %t -- -- -fno-delayed-template-parsing

// CONSTEXPR must stay the first expansion in this file: the crash needs a file
// entry right before it, and a macro or a split '>>' above would hide it.
#define CONSTEXPR constexpr
#define EMPTY_BODY {}

namespace ctor_begins_and_ends_with_macro {
template <typename T>
class CRTP {
public:
    CONSTEXPR CRTP() EMPTY_BODY
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:
    // CHECK-FIXES-NEXT: CONSTEXPR CRTP() EMPTY_BODY
    // CHECK-FIXES-NEXT: public:
    // CHECK-FIXES: friend T;
};

class A : CRTP<A> {};
} // namespace ctor_begins_and_ends_with_macro

#define MEMBER_THEN(X) int I; X
#define THEN_MEMBER(X) X int J;

namespace ctor_in_middle_of_macro {
template <typename T>
class CRTP {
public:
    MEMBER_THEN(CRTP() {})
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: MEMBER_THEN(CRTP() {})
    THEN_MEMBER(CRTP(int) {})
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: THEN_MEMBER(CRTP(int) {})
    // CHECK-FIXES-NOT: friend T;
};

class A : CRTP<A> {};
} // namespace ctor_in_middle_of_macro
