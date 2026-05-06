// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// 
// RUN: %clang_cc1 -std=c++26 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++26 %t/b.cppm -fmodule-file=a=%t/a.pcm -emit-llvm -o - | FileCheck %t/b.cppm

//--- base.hpp
struct base {
	[[__gnu__::__gnu_inline__]] inline ~base() {}
};

//--- a.cppm
module;

#include "base.hpp"

export module a;

export using ::base;

//--- b.cppm
module;
#include "base.hpp"
export module b;
import a;

struct derived : base {
};

void f() {
	derived x;
}

// Just to make sure it won't crash
// CHECK: define void
