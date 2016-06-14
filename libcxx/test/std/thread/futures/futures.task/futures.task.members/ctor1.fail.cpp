//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <future>

// class packaged_task<R(ArgTypes...)>
// template <class F>
//   packaged_task(F&& f);
// These constructors shall not participate in overload resolution if
//    decay<F>::type is the same type as std::packaged_task<R(ArgTypes...)>.

#include <future>
#include <cassert>

struct A {};
typedef std::packaged_task<A(int, char)> PT;
typedef volatile std::packaged_task<A(int, char)> VPT;


int main()
{
    PT p { VPT{} }; // expected-error {{no matching constructor for initialization of 'PT' (aka 'packaged_task<A (int, char)>')}}
    // expected-note@future:* 1 {{candidate template ignored: disabled by 'enable_if'}}
}
