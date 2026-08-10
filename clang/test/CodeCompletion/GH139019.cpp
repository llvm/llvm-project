// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/test.hpp -emit-pch -o %t/1.pch
// RUN: %clang_cc1 -std=c++20 %t/test.cpp -include-pch %t/1.pch -code-completion-at=%t/test.cpp:7:17

//--- test.hpp
#pragma once
class provider_t
{
  public:
    template<class T>
    void emit(T *data)
    {}
};

//--- test.cpp
#include "test.hpp"

void test()
{
    provider_t *focus;
    void *data;
    focus->emit(&data);
}
