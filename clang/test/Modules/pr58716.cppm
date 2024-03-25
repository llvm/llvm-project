// Tests that the compiler won't crash due to the consteval constructor.
//
// REQUIRES: x86-registered-target
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -std=c++20 -emit-module-interface %t/m.cppm -o %t/m.pcm
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -std=c++20 %t/m.pcm -S -emit-llvm -o - | FileCheck %t/m.cppm

//--- m.cppm
module;
#include "fail.h"
export module mymodule;

// CHECK: @.str = {{.*}}"{}\00"
// CHECK: store{{.*}}ptr @.str

//--- fail.h
namespace std { 

template<class _CharT>
class basic_string_view {
public:
    constexpr basic_string_view(const _CharT* __s)
        : __data_(__s) {}

private:
    const   _CharT* __data_;
};

template <class _CharT>
struct basic_format_string {
  template <class _Tp>
  consteval basic_format_string(const _Tp& __str) : __str_{__str} {
  }

private:
  basic_string_view<_CharT> __str_;
};
}

auto this_fails() -> void {
    std::basic_format_string<char> __fmt("{}");
}
