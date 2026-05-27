// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// 
// RUN: %clang_cc1 -std=c++20 %t/counter.cppm -triple x86_64-pc-linux-gnu \
// RUN:   -emit-reduced-module-interface -o %t/counter.pcm
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -triple x86_64-pc-linux-gnu -fprebuilt-module-path=%t \
// RUN:   -disable-llvm-passes -emit-llvm -o - | FileCheck %s

//--- counter.cppm
export module counter;

namespace counter {

// Works without thread_local or with inline keyword
thread_local int next = 1;

export inline auto get_next() noexcept -> int
{
    return next++;
}

}

//--- user.cpp
import counter;

auto user() -> int
{
    return counter::get_next();
}

// CHECK: @_ZN7counterW7counter4nextE = external {{.*}}thread_local {{.*}}global

