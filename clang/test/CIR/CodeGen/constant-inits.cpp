// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct empty{};

struct Point {
    int x;
    int y;
    char c[3];
    int z;
    [[no_unique_address]] empty e;
};

void function() {
    constexpr static empty e;

    constexpr static Point p1{10, 20, {99, 88, 77}, 40, e};

    constexpr static Point array[] {
        {123, 456, {11, 22, 33}, 789, {}},
        {10, 20, {0, 0 ,0}, 40}
    };
}

// CIR: cir.global "private" internal dso_local @_ZZ8functionvE5array = #cir.const_array<[
// CIR-SAME:   #cir.const_record<{#cir.int<123> : !s32i, #cir.int<456> : !s32i, #cir.const_array<[#cir.int<11> : !s8i, #cir.int<22> : !s8i, #cir.int<33> : !s8i]> : !cir.array<!s8i x 3>, #cir.int<789> : !s32i}> : !rec_Point
// CIR-SAME:   #cir.const_record<{#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.zero : !cir.array<!s8i x 3>, #cir.int<40> : !s32i}> : !rec_Point
// CIR-SAME: ]> : !cir.array<!rec_Point x 2>

// CIR: cir.global "private" internal dso_local @_ZZ8functionvE2p1 = #cir.const_record<{#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.const_array<[#cir.int<99> : !s8i, #cir.int<88> : !s8i, #cir.int<77> : !s8i]> : !cir.array<!s8i x 3>, #cir.int<40> : !s32i}> : !rec_Point

// CIR: cir.global "private" internal dso_local @_ZZ8functionvE1e = #cir.zero : !rec_empty

// CIR-LABEL: cir.func dso_local @_Z8functionv()
// CIR:   cir.return


// LLVM-DAG: @_ZZ8functionvE5array = internal global [2 x %struct.Point] [%struct.Point { i32 123, i32 456, [3 x i8] c"\0B\16!", i32 789 }, %struct.Point { i32 10, i32 20, [3 x i8] zeroinitializer, i32 40 }]
// LLVM-DAG: @_ZZ8functionvE2p1 = internal global %struct.Point { i32 10, i32 20, [3 x i8] c"cXM", i32 40 }
// LLVM-DAG: @_ZZ8functionvE1e = internal global %struct.empty zeroinitializer

// LLVM-LABEL: define{{.*}} void @_Z8functionv
// LLVM:   ret void


// OGCG-DAG: @_ZZ8functionvE5array = internal constant [2 x %struct.Point] [%struct.Point { i32 123, i32 456, [3 x i8] c"\0B\16!", i32 789 }, %struct.Point { i32 10, i32 20, [3 x i8] zeroinitializer, i32 40 }]
// OGCG-DAG: @_ZZ8functionvE2p1 = internal constant %struct.Point { i32 10, i32 20, [3 x i8] c"cXM", i32 40 }
// OGCG-DAG: @_ZZ8functionvE1e = internal constant %struct.empty zeroinitializer

// OGCG-LABEL: define{{.*}} void @_Z8functionv
// OGCG:   ret void
