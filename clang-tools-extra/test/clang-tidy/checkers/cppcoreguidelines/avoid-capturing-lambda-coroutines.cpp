// RUN: %check_clang_tidy -std=c++20-or-later %s cppcoreguidelines-avoid-capturing-lambda-coroutines %t -- -- -isystem %S/Inputs/system
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffix=,CPP23 %s cppcoreguidelines-avoid-capturing-lambda-coroutines %t \
// RUN:   -- -config='{CheckOptions: {cppcoreguidelines-avoid-capturing-lambda-coroutines.AllowExplicitObjectParameters: false}}' \
// RUN:   -- -isystem %S/Inputs/system -DCPP23


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
#ifdef CPP23
    [&v](this auto) -> task { co_return; };
    // CHECK-MESSAGES-CPP23: [[@LINE-1]]:5: warning: coroutine lambda may cause use-after-free, avoid captures or ensure lambda closure object has guaranteed lifetime [cppcoreguidelines-avoid-capturing-lambda-coroutines]
#endif
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
}
