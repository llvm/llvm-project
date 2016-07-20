//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// type_traits

// is_callable

// Most testing of is_callable is done within the [meta.trans.other] result_of
// tests.

#include <type_traits>
#include <functional>
#include <memory>

#include "test_macros.h"

struct Tag {};
struct DerFromTag : Tag {};

struct Implicit {
  Implicit(int) {}
};

struct Explicit {
  explicit Explicit(int) {}
};

struct NotCallableWithInt {
  int operator()(int) = delete;
  int operator()(Tag) { return 42; }
};

int main()
{
    {
        using Fn = int(Tag::*)(int);
        using RFn = int(Tag::*)(int) &&;
        // INVOKE bullet 1, 2 and 3
        {
            // Bullet 1
            static_assert(std::is_callable<Fn(Tag&, int)>::value, "");
            static_assert(std::is_callable<Fn(DerFromTag&, int)>::value, "");
            static_assert(std::is_callable<RFn(Tag&&, int)>::value, "");
            static_assert(!std::is_callable<RFn(Tag&, int)>::value, "");
            static_assert(!std::is_callable<Fn(Tag&)>::value, "");
            static_assert(!std::is_callable<Fn(Tag const&, int)>::value, "");
        }
        {
            // Bullet 2
            using T = std::reference_wrapper<Tag>;
            using DT = std::reference_wrapper<DerFromTag>;
            using CT = std::reference_wrapper<const Tag>;
            static_assert(std::is_callable<Fn(T&, int)>::value, "");
            static_assert(std::is_callable<Fn(DT&, int)>::value, "");
            static_assert(std::is_callable<Fn(const T&, int)>::value, "");
            static_assert(std::is_callable<Fn(T&&, int)>::value, "");
            static_assert(!std::is_callable<Fn(CT&, int)>::value, "");
            static_assert(!std::is_callable<RFn(T, int)>::value, "");
        }
        {
            // Bullet 3
            using T = Tag*;
            using DT = DerFromTag*;
            using CT = const Tag*;
            using ST = std::unique_ptr<Tag>;
            static_assert(std::is_callable<Fn(T&, int)>::value, "");
            static_assert(std::is_callable<Fn(DT&, int)>::value, "");
            static_assert(std::is_callable<Fn(const T&, int)>::value, "");
            static_assert(std::is_callable<Fn(T&&, int)>::value, "");
            static_assert(std::is_callable<Fn(ST, int)>::value, "");
            static_assert(!std::is_callable<Fn(CT&, int)>::value, "");
            static_assert(!std::is_callable<RFn(T, int)>::value, "");
        }
    }
    {
        // Bullets 4, 5 and 6
        using Fn = int (Tag::*);
        static_assert(!std::is_callable<Fn()>::value, "");
        {
            // Bullet 4
            static_assert(std::is_callable<Fn(Tag&)>::value, "");
            static_assert(std::is_callable<Fn(DerFromTag&)>::value, "");
            static_assert(std::is_callable<Fn(Tag&&)>::value, "");
            static_assert(std::is_callable<Fn(Tag const&)>::value, "");
        }
        {
            // Bullet 5
            using T = std::reference_wrapper<Tag>;
            using DT = std::reference_wrapper<DerFromTag>;
            using CT = std::reference_wrapper<const Tag>;
            static_assert(std::is_callable<Fn(T&)>::value, "");
            static_assert(std::is_callable<Fn(DT&)>::value, "");
            static_assert(std::is_callable<Fn(const T&)>::value, "");
            static_assert(std::is_callable<Fn(T&&)>::value, "");
            static_assert(std::is_callable<Fn(CT&)>::value, "");
        }
        {
            // Bullet 6
            using T = Tag*;
            using DT = DerFromTag*;
            using CT = const Tag*;
            using ST = std::unique_ptr<Tag>;
            static_assert(std::is_callable<Fn(T&)>::value, "");
            static_assert(std::is_callable<Fn(DT&)>::value, "");
            static_assert(std::is_callable<Fn(const T&)>::value, "");
            static_assert(std::is_callable<Fn(T&&)>::value, "");
            static_assert(std::is_callable<Fn(ST)>::value, "");
            static_assert(std::is_callable<Fn(CT&)>::value, "");
        }
    }
    {
        // INVOKE bullet 7
        {
            // Function pointer
            using Fp = void(*)(Tag&, int);
            static_assert(std::is_callable<Fp(Tag&, int)>::value, "");
            static_assert(std::is_callable<Fp(DerFromTag&, int)>::value, "");
            static_assert(!std::is_callable<Fp(const Tag&, int)>::value, "");
            static_assert(!std::is_callable<Fp()>::value, "");
            static_assert(!std::is_callable<Fp(Tag&)>::value, "");
        }
        {
            // Function reference
            using Fp = void(&)(Tag&, int);
            static_assert(std::is_callable<Fp(Tag&, int)>::value, "");
            static_assert(std::is_callable<Fp(DerFromTag&, int)>::value, "");
            static_assert(!std::is_callable<Fp(const Tag&, int)>::value, "");
            static_assert(!std::is_callable<Fp()>::value, "");
            static_assert(!std::is_callable<Fp(Tag&)>::value, "");
        }
        {
            // Function object
            using Fn = NotCallableWithInt;
            static_assert(std::is_callable<Fn(Tag)>::value, "");
            static_assert(!std::is_callable<Fn(int)>::value, "");
        }
    }
    {
        // Check that the conversion to the return type is properly checked
        using Fn = int(*)();
        static_assert(std::is_callable<Fn(), Implicit>::value, "");
        static_assert(std::is_callable<Fn(), double>::value, "");
        static_assert(std::is_callable<Fn(), const volatile void>::value, "");
        static_assert(!std::is_callable<Fn(), Explicit>::value, "");
    }
    {
        // Check for is_callable_v
        using Fn = void(*)();
        static_assert(std::is_callable_v<Fn()>, "");
        static_assert(!std::is_callable_v<Fn(int)>, "");
    }
}
