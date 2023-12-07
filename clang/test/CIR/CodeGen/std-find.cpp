// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -clangir-disable-emit-cxx-default -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

// CHECK: ![[array:.*]] = !cir.struct<struct "std::array<unsigned char, 9>"

int test_find(unsigned char n = 3)
{
    // CHECK: cir.func @_Z9test_findh(%arg0: !u8i
    unsigned num_found = 0;
    std::array<unsigned char, 9> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    // CHECK: %[[array_addr:.*]] = cir.alloca ![[array]], cir.ptr <![[array]]>, ["v"]

    auto f = std::find(v.begin(), v.end(), n);
    // CHECK: {{.*}} cir.call @_ZNSt5arrayIhLj9EE5beginEv(%[[array_addr]])
    // CHECK: {{.*}} cir.call @_ZNSt5arrayIhLj9EE3endEv(%[[array_addr]])
    // CHECK: {{.*}} cir.call @_ZSt4findINSt5arrayIhLj9EE8iteratorEhET_S3_S3_RKT0_(

    if (f != v.end())
        num_found++;
    // CHECK: {{.*}} cir.call @_ZNSt5arrayIhLj9EE3endEv(%[[array_addr]]
    // CHECK: %[[neq_cmp:.*]] = cir.call @_ZNSt5arrayIhLj9EE8iteratorneES1_(
    // CHECK: cir.if %[[neq_cmp]]

    return num_found;
}