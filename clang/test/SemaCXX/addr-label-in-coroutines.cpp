// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

#include "Inputs/std-coroutine.h"

struct resumable {
  struct promise_type {
    resumable get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
  };
};

resumable f1(int &out, int *inst) {
    static void* dispatch_table[] = {&&inc,      // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
                                     &&suspend,  // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
                                     &&stop};    // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
    #define DISPATCH() goto *dispatch_table[*inst++]
inc:
    out++;
    DISPATCH();

suspend:
    co_await std::suspend_always{};
    DISPATCH();

stop:
    co_return;
}

resumable f2(int &out, int *inst) {
    void* dispatch_table[] = {nullptr, nullptr, nullptr};
    dispatch_table[0] = &&inc;      // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
    dispatch_table[1] = &&suspend;  // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
    dispatch_table[2] = &&stop;     // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
    #define DISPATCH() goto *dispatch_table[*inst++]
inc:
    out++;
    DISPATCH();

suspend:
    co_await std::suspend_always{};
    DISPATCH();

stop:
    co_return;
}

resumable f3(int &out, int *inst) {
    void* dispatch_table[] = {nullptr, nullptr, nullptr};
    [&]() -> resumable {
        dispatch_table[0] = &&inc;      // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
        dispatch_table[1] = &&suspend;  // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
        dispatch_table[2] = &&stop;     // expected-error {{the GNU address of label extension is not allowed in coroutines.}}
        #define DISPATCH() goto *dispatch_table[*inst++]
    inc:
        out++;
        DISPATCH();

    suspend:
        co_await std::suspend_always{};
        DISPATCH();

    stop:
        co_return;
    }();

    co_return;
}
