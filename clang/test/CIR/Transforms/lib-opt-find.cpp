// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -clangir-disable-emit-cxx-default -fclangir -fclangir-idiom-recognizer -fclangir-lib-opt -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

int test1(unsigned char n = 3)
{
    // CHECK: test1
    unsigned num_found = 0;
    // CHECK: %[[pattern_addr:.*]] = cir.alloca !u8i, cir.ptr <!u8i>, ["n"
    std::array<unsigned char, 9> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto f = std::find(v.begin(), v.end(), n);

    // CHECK: %[[first:.*]] = cir.call @_ZNSt5arrayIhLj9EE5beginEv
    // CHECK: %[[last:.*]] = cir.call @_ZNSt5arrayIhLj9EE3endEv
    // CHECK: %[[cast_to_void:.*]] = cir.cast(bitcast, %[[first]] : !cir.ptr<!u8i>), !cir.ptr<!void>
    // CHECK: %[[load_pattern:.*]] = cir.load %[[pattern_addr]] : cir.ptr <!u8i>, !u8i
    // CHECK: %[[pattern:.*]] = cir.cast(integral, %[[load_pattern:.*]] : !u8i), !s32i

    // CHECK-NOT: {{.*}} cir.call @_ZSt4findIPhhET_S1_S1_RKT0_(
    // CHECK: %[[array_size:.*]] = cir.const(#cir.int<9> : !u64i) : !u64i

    // CHECK: %[[result_cast:.*]] = cir.libc.memchr(%[[cast_to_void]], %[[pattern]], %[[array_size]])
    // CHECK: %[[memchr_res:.*]] = cir.cast(bitcast, %[[result_cast]] : !cir.ptr<!void>), !cir.ptr<!u8i>
    // CHECK: %[[nullptr:.*]] = cir.const(#cir.ptr<null> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
    // CHECK: %[[cmp_res:.*]] = cir.cmp(eq, %[[nullptr]], %[[memchr_res]]) : !cir.ptr<!u8i>, !cir.bool
    // CHECK: cir.ternary(%[[cmp_res]], true {
    // CHECK:   cir.yield %[[last]] : !cir.ptr<!u8i>
    // CHECK: }, false {
    // CHECK:   cir.yield %[[memchr_res]] : !cir.ptr<!u8i>
    // CHECK: }) : (!cir.bool) -> !cir.ptr<!u8i>

    if (f != v.end())
        num_found++;

    return num_found;
}

unsigned char* test2(unsigned char* first, unsigned char* last, unsigned char v)
{
    return std::find(first, last, v);
    // CHECK: test2

    // CHECK: %[[first_storage:.*]] = cir.alloca !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>, ["first", init]
    // CHECK: %[[last_storage:.*]] = cir.alloca !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>, ["last", init]
    // CHECK: %[[pattern_storage:.*]] = cir.alloca !u8i, cir.ptr <!u8i>, ["v", init]
    // CHECK: %[[first:.*]] = cir.load %[[first_storage]]
    // CHECK: %[[last:.*]] = cir.load %[[last_storage]]
    // CHECK: %[[cast_to_void:.*]] = cir.cast(bitcast, %[[first]] : !cir.ptr<!u8i>), !cir.ptr<!void>
    // CHECK: %[[load_pattern:.*]] = cir.load %[[pattern_storage]] : cir.ptr <!u8i>, !u8i
    // CHECK: %[[pattern:.*]] = cir.cast(integral, %[[load_pattern:.*]] : !u8i), !s32i

    // CHECK-NOT: {{.*}} cir.call @_ZSt4findIPhhET_S1_S1_RKT0_(
    // CHECK: %[[array_size:.*]] = cir.ptr_diff(%[[last]], %[[first]]) : !cir.ptr<!u8i> -> !u64i

    // CHECK: %[[result_cast:.*]] = cir.libc.memchr(%[[cast_to_void]], %[[pattern]], %[[array_size]])
    // CHECK: %[[memchr_res:.*]] = cir.cast(bitcast, %[[result_cast]] : !cir.ptr<!void>), !cir.ptr<!u8i>
    // CHECK: %[[nullptr:.*]] = cir.const(#cir.ptr<null> : !cir.ptr<!u8i>) : !cir.ptr<!u8i>
    // CHECK: %[[cmp_res:.*]] = cir.cmp(eq, %[[nullptr]], %[[memchr_res]]) : !cir.ptr<!u8i>, !cir.bool
    // CHECK: cir.ternary(%[[cmp_res]], true {
    // CHECK:   cir.yield %[[last]] : !cir.ptr<!u8i>
    // CHECK: }, false {
    // CHECK:   cir.yield %[[memchr_res]] : !cir.ptr<!u8i>
    // CHECK: }) : (!cir.bool) -> !cir.ptr<!u8i>
}
