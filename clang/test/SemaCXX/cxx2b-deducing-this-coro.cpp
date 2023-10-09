// RUN: %clang_cc1 -std=c++2b %s -fsyntax-only -verify

#include "Inputs/std-coroutine.h"

struct S;
template <typename T>
class coro_test {
public:
    struct promise_type;
    using handle = std::coroutine_handle<promise_type>;
	struct promise_type {
        promise_type(const promise_type&) = delete; // #copy-ctr
        promise_type(T);  // #candidate
        coro_test get_return_object();
        std::suspend_never initial_suspend();
	    std::suspend_never final_suspend() noexcept;
	    void return_void();
        void unhandled_exception();


        template<typename Arg, typename... Args>
        void* operator new(decltype(0zu) sz, Arg&&, Args&... args) {
            static_assert(!__is_same(__decay(Arg), S), "Ok"); // expected-error 2{{Ok}}
        }

    };
private:
	handle h;
};


template <typename Ret, typename... P>
struct std::coroutine_traits<coro_test<S&>, Ret, P...> {
  using promise_type = coro_test<S&>::promise_type;
  static_assert(!__is_same(Ret, S&), "Ok"); // expected-error{{static assertion failed due to requirement '!__is_same(S &, S &)': Ok}}
};


struct S {

    coro_test<S&> ok(this S&, int) {
        co_return; // expected-note {{in instantiation}}
    }

    coro_test<const S&> ok2(this const S&) { // expected-note {{in instantiation}}
        co_return;
    }

    coro_test<int> ko(this const S&) {  // expected-error {{no matching constructor for initialization of 'std::coroutine_traits<coro_test<int>, const S &>::promise_type'}} \
                                        // expected-note {{in instantiation}} \
                                       // FIXME: the message below is unhelpful but this is pre-existing
                                       // expected-note@#candidate {{candidate constructor not viable: requires 1 argument, but 0 were provided}} \
                                       // expected-note@#copy-ctr  {{candidate constructor not viable}}
        co_return;
    }

};
