// RUN: %check_clang_tidy -std=c++20-or-later %s cppcoreguidelines-avoid-capturing-lambda-coroutines %t -- -- -isystem %S/Inputs/system

#include <coroutines.h>

void Caught() {
    int v;

    [&] () -> task { int y = v; co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [=] () -> task { int y = v; co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [v] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [&v] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [y=v] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    [y{v}] () -> task { co_return; };
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
}

struct S {
    void m() {
        [this] () -> task { co_return; };
        // CHECK-MESSAGES: [[@LINE-1]]:9: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
    }
};

void Safe() {
    int v;
    [] () -> task { co_return; };
    [&] () -> task { co_return; };
    [=] () -> task { co_return; };

    [&v]{++v;}();

#if __cplusplus >= 202302L
    // Lambda coroutines using C++23 deducing this are safe.
    [&v] (this auto) -> task { co_return; };
    [v] (this auto, int x) -> task { co_return; };
#endif
}
