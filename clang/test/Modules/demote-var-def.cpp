// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// DEFINE: %{common-flags}= -I %t -isystem %t -xc++ -std=c++20 -fmodules
//
// RUN: mkdir -p %t/b2
// RUN: mkdir -p %t/b1
// RUN: %clang_cc1 %{common-flags} -emit-module -fmodule-name=module_d \
// RUN:     d.cppmap -o d.pcm
// RUN: %clang_cc1 %{common-flags} -emit-module -fmodule-name=module_a \
// RUN:     -fmodule-file=d.pcm  a.cppmap -o a.pcm
// RUN: %clang_cc1 %{common-flags} -emit-module -fmodule-name=module_b2 \
// RUN:     -fmodule-file=a.pcm b2/b.cppmap -o b2/b.pcm
// RUN: %clang_cc1 %{common-flags} -emit-module -fmodule-name=module_b1 \
// RUN:     -fmodule-file=b2/b.pcm b1/b.cppmap -o b1/b.pcm
// RUN: %clang_cc1 %{common-flags} -emit-module -fmodule-name=module_f \
// RUN:     -fmodule-file=b1/b.pcm f.cppmap -o f.pcm
// RUN: %clang_cc1 %{common-flags} -emit-module -fmodule-name=module_c \
// RUN:     -fmodule-file=f.pcm c.cppmap -o c.pcm
// RUN: %clang_cc1 %{common-flags} -emit-module \
// RUN:     -fmodule-name=module_e e.cppmap -o e.pcm
//
// RUN: %clang_cc1 %{common-flags} \
// RUN:     -fmodule-file=c.pcm -fmodule-file=e.pcm \
// RUN:     src.cpp -o src.pic.o

//--- invoke.h
#ifndef _LIBCPP___TYPE_TRAITS_IS_SAME_H
#define _LIBCPP___TYPE_TRAITS_IS_SAME_H
namespace std { inline namespace _LIBCPP_ABI_NAMESPACE {
template <class _Tp, class _Up>
constexpr bool is_same_v = __is_same(_Tp, _Up);
} }
#endif

//--- memory
#include <invoke.h>
namespace std { inline namespace _LIBCPP_ABI_NAMESPACE {
template <class _Tp>
using __decay_t = __decay(_Tp);
template <class _Tp>
using decay_t = __decay_t<_Tp>;
} }

//--- other.h
#include <invoke.h>

//--- a.cppmap
module "module_a" {
}

//--- b1/b.cppmap
module "module_b1" {
}

//--- b2/b.cppmap
module "module_b2" {
}

//--- c.cppmap
module "module_c" {
}

//--- d.cppmap
module "module_d" {
    header "d.h"
}

//--- d.h
#include <other.h>

//--- e.cppmap
module "module_e" {
    header "e.h"
}

//--- e.h
#include <memory>

//--- f.cppmap
module "module_f" {
}

//--- src.cpp
#include <d.h>
#include <memory>
template <typename T>
concept coroutine_result =
    std::is_same_v<std::decay_t<T>, T>;
template <coroutine_result R>
class Co;
using T = Co<void>;
