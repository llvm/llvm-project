// RUN: %check_clang_tidy -std=c++20-or-later %s cppcoreguidelines-avoid-capturing-lambda-coroutines %t -- -- \
// RUN:   -isystem %S/../readability/Inputs/identifier-naming/system

#include <coroutines.h>

void Caught() {
    int v;

    [&] () -> task { int y = v; co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: found capturing coroutine lambda [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [=] () -> task { int y = v; co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: found capturing coroutine lambda [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [v] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: found capturing coroutine lambda [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [&v] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: found capturing coroutine lambda [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [y=v] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: found capturing coroutine lambda [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [y{v}] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: found capturing coroutine lambda [cppcoreguidelines-avoid-capturing-lambda-coroutines]
}

struct S {
    void m() {
        [this] () -> task { co_return; };
        // CHECK-MESSAGES: [[@LINE-1]]:9: warning: found capturing coroutine lambda [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    }
};

void Safe() {
    int v;
    [] () -> task { co_return; };
    [&] () -> task { co_return; };
    [=] () -> task { co_return; };
}
