//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template<CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);
// template<Returnable R, CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);

#include <functional>
#include <cassert>

int count = 0;

template <class F>
void
test(F f)
{
    int save_count = count;
    f();
    assert(count == save_count + 1);
}

template <class F>
void
test_const(const F& f)
{
    int save_count = count;
    f();
    assert(count == save_count + 2);
}

void f() {++count;}

struct A_int_0
{
    void operator()() {++count;}
    void operator()() const {count += 2;}
};

int main()
{
    test(std::bind(f));
    test(std::bind(&f));
    test(std::bind(A_int_0()));
    test_const(std::bind(A_int_0()));

    test(std::bind<void>(f));
    test(std::bind<void>(&f));
    test(std::bind<void>(A_int_0()));
    test_const(std::bind<void>(A_int_0()));
}
