// Address: https://github.com/llvm/llvm-project/issues/60275
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/b.cpp -fmodule-file=%t/a.pcm -emit-llvm -o - | FileCheck %t/b.cpp
//--- foo.h

consteval void global() {}

//--- a.cppm
module;
#include "foo.h"
export module a;

//--- b.cpp
#include "foo.h"
import a;

consteval int b() {
	return 0;
}

struct bb {
	int m = b();
};

void bbb() {
	bb x;
}

// CHECK: define{{.*}}_ZN2bbC2Ev({{.*}}[[THIS:%.+]])
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[THIS_ADDR:%.*]] = alloca ptr
// CHECK-NEXT:   store ptr [[THIS]], ptr [[THIS_ADDR]]
// CHECK-NEXT:   [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]]
// CHECK-NEXT:   [[M_ADDR:%.*]] = getelementptr{{.*}}%struct.bb, ptr [[THIS1]], i32 0, i32 0
// CHECK-NEXT:   store i32 0, ptr [[M_ADDR]]
