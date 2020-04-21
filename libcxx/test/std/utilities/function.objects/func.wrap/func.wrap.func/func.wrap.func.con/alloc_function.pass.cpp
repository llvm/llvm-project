//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// REQUIRES: c++98 || c++03 || c++11 || c++14

// class function<R(ArgTypes...)>

// template<class A> function(allocator_arg_t, const A&, const function&);

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "count_new.h"
#include "../function_types.h"

class DummyClass {};

template <class FuncType, class AllocType>
void test_FunctionObject(AllocType& alloc)
{
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    // Construct function from FunctionObject.
    std::function<FuncType> f = FunctionObject();
    assert(FunctionObject::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(f.template target<FunctionObject>());
    assert(f.template target<FuncType>() == 0);
    assert(f.template target<FuncType*>() == 0);
    // Copy function with allocator
    std::function<FuncType> f2(std::allocator_arg, alloc, f);
    assert(FunctionObject::count == 2);
    assert(globalMemCounter.checkOutstandingNewEq(2));
    assert(f2.template target<FunctionObject>());
    assert(f2.template target<FuncType>() == 0);
    assert(f2.template target<FuncType*>() == 0);
    }
    assert(FunctionObject::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
}

template <class FuncType, class AllocType>
void test_FreeFunction(AllocType& alloc)
{
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    // Construct function from function pointer.
    FuncType* target = &FreeFunction;
    std::function<FuncType> f = target;
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(f.template target<FuncType*>());
    assert(*f.template target<FuncType*>() == target);
    assert(f.template target<FuncType>() == 0);
    // Copy function with allocator
    std::function<FuncType> f2(std::allocator_arg, alloc, f);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(f2.template target<FuncType*>());
    assert(*f2.template target<FuncType*>() == target);
    assert(f2.template target<FuncType>() == 0);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
}

template <class TargetType, class FuncType, class AllocType>
void test_MemFunClass(AllocType& alloc)
{
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    // Construct function from function pointer.
    TargetType target = &MemFunClass::foo;
    std::function<FuncType> f = target;
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(f.template target<TargetType>());
    assert(*f.template target<TargetType>() == target);
    assert(f.template target<FuncType*>() == 0);
    // Copy function with allocator
    std::function<FuncType> f2(std::allocator_arg, alloc, f);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(f2.template target<TargetType>());
    assert(*f2.template target<TargetType>() == target);
    assert(f2.template target<FuncType*>() == 0);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
}

template <class Alloc>
void test_for_alloc(Alloc& alloc)
{
    // Large FunctionObject -- Allocation should occur
    test_FunctionObject<int()>(alloc);
    test_FunctionObject<int(int)>(alloc);
    test_FunctionObject<int(int, int)>(alloc);
    test_FunctionObject<int(int, int, int)>(alloc);
    // Free function -- No allocation should occur
    test_FreeFunction<int()>(alloc);
    test_FreeFunction<int(int)>(alloc);
    test_FreeFunction<int(int, int)>(alloc);
    test_FreeFunction<int(int, int, int)>(alloc);
    // Member function -- No allocation should occur.
    test_MemFunClass<int(MemFunClass::*)() const, int(MemFunClass&)>(alloc);
    test_MemFunClass<int(MemFunClass::*)(int) const, int(MemFunClass&, int)>(alloc);
    test_MemFunClass<int(MemFunClass::*)(int, int) const, int(MemFunClass&, int, int)>(alloc);
}

int main(int, char**)
{
  globalMemCounter.reset();
  {
    bare_allocator<DummyClass> alloc;
    test_for_alloc(alloc);
  }
  {
    non_default_test_allocator<DummyClass> alloc(42);
    test_for_alloc(alloc);
  }

  return 0;
}
