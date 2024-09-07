// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify
#include "Inputs/std-coroutine.h"

namespace std {
typedef decltype(sizeof(int)) size_t;
}

struct Allocator {};

struct resumable {
  struct promise_type {
    void *operator new(std::size_t sz, Allocator &);

    resumable get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
  };
};

resumable f1() { // expected-error {{'operator new' provided by 'std::coroutine_traits<resumable>::promise_type' (aka 'resumable::promise_type') is not usable with the function signature of 'f1'}}
  co_return;
}

// NOTE: Although the argument here is a rvalue reference and the corresponding
// allocation function in resumable::promise_type have lvalue references, it looks
// the signature of f2 is invalid. But according to [dcl.fct.def.coroutine]p4:
//
//   In the following, pi is an lvalue of type Pi, where p1 denotes the object 
//   parameter and pi+1 denotes the ith non-object function parameter for a
//   non-static member function.
//
// And [dcl.fct.def.coroutine]p9.1
//
//   overload resolution is performed on a function call created by assembling an argument list.
//   The first argument is the amount of space requested, and has type std::size_­t.
//   The lvalues p1…pn are the succeeding arguments.
//
// So the actual type passed to resumable::promise_type::operator new is lvalue
// Allocator. It is allowed  to convert a lvalue to a lvalue reference. So the 
// following one is valid.
resumable f2(Allocator &&) {
  co_return;
}

resumable f3(Allocator &) {
  co_return;
}

resumable f4(Allocator) {
  co_return;
}

resumable f5(const Allocator) { // expected-error {{operator new' provided by 'std::coroutine_traits<resumable, const Allocator>::promise_type' (aka 'resumable::promise_type') is not usable}}
  co_return;
}

resumable f6(const Allocator &) { // expected-error {{operator new' provided by 'std::coroutine_traits<resumable, const Allocator &>::promise_type' (aka 'resumable::promise_type') is not usable}}
  co_return;
}

struct promise_base1 {
  void *operator new(std::size_t sz); // expected-note {{member found by ambiguous name lookup}}
};

struct promise_base2 {
  void *operator new(std::size_t sz); // expected-note {{member found by ambiguous name lookup}}
};

struct resumable2 {
  struct promise_type : public promise_base1, public promise_base2 {
    resumable2 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
  };
};

resumable2 f7() { // expected-error {{member 'operator new' found in multiple base classes of different types}}
  co_return;
}
