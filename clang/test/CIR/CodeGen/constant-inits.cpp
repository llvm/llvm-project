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

struct [[gnu::packed]] packed {
    char c;
    int i;
};

struct [[gnu::packed]] alignas(8) packed_and_aligned {
    short s;
    char c;
    float f;
};

struct simple {
    int a, b;
};

// Byte-aligned bitfields
struct byte_aligned_bitfields {
    unsigned int a : 8;
    unsigned int b : 8;
    unsigned int c : 16;
};

struct signed_byte_aligned_bitfields {
    int x : 8;
    int y : 8;
};

struct single_byte_bitfield {
    unsigned char a : 8;
};

// Partial bitfields (sub-byte)
struct partial_bitfields {
    unsigned int a : 3;
    unsigned int b : 5;
    unsigned int c : 8;
};

struct signed_partial_bitfields {
    int x : 4;
    int y : 4;
};

struct mixed_partial_bitfields {
    unsigned char a : 1;
    unsigned char b : 1;
    unsigned char c : 1;
    unsigned char d : 5;
};

void function() {
    constexpr static empty e;

    constexpr static Point p1{10, 20, {99, 88, 77}, 40, e};

    constexpr static Point array[] {
        {123, 456, {11, 22, 33}, 789, {}},
        {10, 20, {0, 0 ,0}, 40}
    };

    constexpr static packed p2 {123, 456};
    constexpr static packed packed_array[] {
        p2, p2
    };

    constexpr static packed_and_aligned paa {1, 2, 3.f};
    constexpr static packed_and_aligned paa_array[2] {
        {1, 2, 3.f}
    };

    constexpr static simple s{0, -1};
    constexpr static simple simple_array[] {
        s, {1111, 2222}, s
    };

    // Byte-aligned bitfield tests
    constexpr static byte_aligned_bitfields ba_bf1 = {0xFF, 0xAA, 0x1234};
    constexpr static signed_byte_aligned_bitfields ba_bf2 = {-1, 127};
    constexpr static single_byte_bitfield ba_bf3 = {42};

    // Partial bitfield tests
    constexpr static partial_bitfields p_bf1 = {1, 2, 3};
    constexpr static signed_partial_bitfields p_bf2 = {-1, 7};
    constexpr static mixed_partial_bitfields p_bf3 = {1, 0, 1, 15};
}

// Anonymous struct type definitions for bitfields
// CIR-DAG: !rec_anon_struct = !cir.record<struct  {!u8i, !u8i, !u8i, !u8i}>
// CIR-DAG: !rec_anon_struct1 = !cir.record<struct  {!u8i, !u8i, !cir.array<!u8i x 2>}>

// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE1e = #cir.zero : !rec_empty
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE1s = #cir.const_record<{#cir.int<0> : !s32i, #cir.int<-1> : !s32i}> : !rec_simple
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE2p1 = #cir.const_record<{#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.const_array<[#cir.int<99> : !s8i, #cir.int<88> : !s8i, #cir.int<77> : !s8i]> : !cir.array<!s8i x 3>, #cir.int<40> : !s32i}> : !rec_Point
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE2p2 = #cir.const_record<{#cir.int<123> : !s8i, #cir.int<456> : !s32i}> : !rec_packed
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE3paa = #cir.const_record<{#cir.int<1> : !s16i, #cir.int<2> : !s8i, #cir.fp<3.000000e+00> : !cir.float, #cir.zero : !u8i}> : !rec_packed_and_aligned

// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE5array = #cir.const_array<[
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<123> : !s32i, #cir.int<456> : !s32i, #cir.const_array<[#cir.int<11> : !s8i, #cir.int<22> : !s8i, #cir.int<33> : !s8i]> : !cir.array<!s8i x 3>, #cir.int<789> : !s32i}> : !rec_Point
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.zero : !cir.array<!s8i x 3>, #cir.int<40> : !s32i}> : !rec_Point
// CIR-DAG-SAME: ]> : !cir.array<!rec_Point x 2>

// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE12simple_array = #cir.const_array<[
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<0> : !s32i, #cir.int<-1> : !s32i}> : !rec_simple,
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<1111> : !s32i, #cir.int<2222> : !s32i}> : !rec_simple,
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<0> : !s32i, #cir.int<-1> : !s32i}> : !rec_simple
// CIR-DAG-SAME: ]> : !cir.array<!rec_simple x 3>

// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE12packed_array = #cir.const_array<[
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<123> : !s8i, #cir.int<456> : !s32i}> : !rec_packed,
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<123> : !s8i, #cir.int<456> : !s32i}> : !rec_packed
// CIR-DAG-SAME: ]> : !cir.array<!rec_packed x 2>

// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE9paa_array = #cir.const_array<[
// CIR-DAG-SAME:   #cir.const_record<{#cir.int<1> : !s16i, #cir.int<2> : !s8i, #cir.fp<3.000000e+00> : !cir.float, #cir.zero : !u8i}> : !rec_packed_and_aligned,
// CIR-DAG-SAME:   #cir.zero : !rec_packed_and_aligned
// CIR-DAG-SAME: ]> : !cir.array<!rec_packed_and_aligned x 2>

// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE6ba_bf1 = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<255> : !u8i,
// CIR-DAG-SAME:   #cir.int<170> : !u8i,
// CIR-DAG-SAME:   #cir.int<52> : !u8i,
// CIR-DAG-SAME:   #cir.int<18> : !u8i
// CIR-DAG-SAME: }> : !rec_anon_struct
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE6ba_bf2 = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<255> : !u8i,
// CIR-DAG-SAME:   #cir.int<127> : !u8i,
// CIR-DAG-SAME:   #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 2>
// CIR-DAG-SAME: }> : !rec_anon_struct1
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE6ba_bf3 = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<42> : !u8i
// CIR-DAG-SAME: }> : !rec_single_byte_bitfield
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE5p_bf1 = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<17> : !u8i,
// CIR-DAG-SAME:   #cir.int<3> : !u8i,
// CIR-DAG-SAME:   #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 2>
// CIR-DAG-SAME: }> : !rec_anon_struct1
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE5p_bf2 = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<127> : !u8i,
// CIR-DAG-SAME:   #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>
// CIR-DAG-SAME: }> : !rec_signed_partial_bitfields
// CIR-DAG: cir.global "private" internal dso_local @_ZZ8functionvE5p_bf3 = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<125> : !u8i
// CIR-DAG-SAME: }> : !rec_mixed_partial_bitfields

// CIR-LABEL: cir.func dso_local @_Z8functionv()
// CIR:   cir.return


// LLVM-DAG: @_ZZ8functionvE12packed_array = internal global [2 x %struct.packed] [%struct.packed <{ i8 123, i32 456 }>, %struct.packed <{ i8 123, i32 456 }>]
// LLVM-DAG: @_ZZ8functionvE12simple_array = internal global [3 x %struct.simple] [%struct.simple { i32 0, i32 -1 }, %struct.simple { i32 1111, i32 2222 }, %struct.simple { i32 0, i32 -1 }]
// LLVM-DAG: @_ZZ8functionvE1e = internal global %struct.empty zeroinitializer
// LLVM-DAG: @_ZZ8functionvE1s = internal global %struct.simple { i32 0, i32 -1 }
// LLVM-DAG: @_ZZ8functionvE2p1 = internal global %struct.Point { i32 10, i32 20, [3 x i8] c"cXM", i32 40 }
// LLVM-DAG: @_ZZ8functionvE2p2 = internal global %struct.packed <{ i8 123, i32 456 }>
// LLVM-DAG: @_ZZ8functionvE3paa = internal global %struct.packed_and_aligned <{ i16 1, i8 2, float 3.000000e+00, i8 0 }>
// LLVM-DAG: @_ZZ8functionvE5array = internal global [2 x %struct.Point] [%struct.Point { i32 123, i32 456, [3 x i8] c"\0B\16!", i32 789 }, %struct.Point { i32 10, i32 20, [3 x i8] zeroinitializer, i32 40 }]
// LLVM-DAG: @_ZZ8functionvE9paa_array = internal global [2 x %struct.packed_and_aligned] [%struct.packed_and_aligned <{ i16 1, i8 2, float 3.000000e+00, i8 0 }>, %struct.packed_and_aligned zeroinitializer]
// LLVM-DAG: @_ZZ8functionvE6ba_bf1 = internal global { i8, i8, i8, i8 } { i8 -1, i8 -86, i8 52, i8 18 }
// LLVM-DAG: @_ZZ8functionvE6ba_bf2 = internal global { i8, i8, [2 x i8] } { i8 -1, i8 127, [2 x i8] zeroinitializer }
// LLVM-DAG: @_ZZ8functionvE6ba_bf3 = internal global %struct.single_byte_bitfield { i8 42 }
// LLVM-DAG: @_ZZ8functionvE5p_bf1 = internal global { i8, i8, [2 x i8] } { i8 17, i8 3, [2 x i8] zeroinitializer }
// LLVM-DAG: @_ZZ8functionvE5p_bf2 = internal global %struct.signed_partial_bitfields { i8 127, [3 x i8] zeroinitializer }
// LLVM-DAG: @_ZZ8functionvE5p_bf3 = internal global %struct.mixed_partial_bitfields { i8 125 }

// LLVM-LABEL: define{{.*}} void @_Z8functionv
// LLVM:   ret void


// OGCG-DAG: @_ZZ8functionvE12packed_array = internal constant [2 x %struct.packed] [%struct.packed <{ i8 123, i32 456 }>, %struct.packed <{ i8 123, i32 456 }>]
// OGCG-DAG: @_ZZ8functionvE12simple_array = internal constant [3 x %struct.simple] [%struct.simple { i32 0, i32 -1 }, %struct.simple { i32 1111, i32 2222 }, %struct.simple { i32 0, i32 -1 }]
// OGCG-DAG: @_ZZ8functionvE1e = internal constant %struct.empty zeroinitializer
// OGCG-DAG: @_ZZ8functionvE1s = internal constant %struct.simple { i32 0, i32 -1 }
// OGCG-DAG: @_ZZ8functionvE2p1 = internal constant %struct.Point { i32 10, i32 20, [3 x i8] c"cXM", i32 40 }
// OGCG-DAG: @_ZZ8functionvE2p2 = internal constant %struct.packed <{ i8 123, i32 456 }>
// OGCG-DAG: @_ZZ8functionvE3paa = internal constant %struct.packed_and_aligned <{ i16 1, i8 2, float 3.000000e+00, i8 undef }>
// OGCG-DAG: @_ZZ8functionvE5array = internal constant [2 x %struct.Point] [%struct.Point { i32 123, i32 456, [3 x i8] c"\0B\16!", i32 789 }, %struct.Point { i32 10, i32 20, [3 x i8] zeroinitializer, i32 40 }]
// OGCG-DAG: @_ZZ8functionvE9paa_array = internal constant [2 x %struct.packed_and_aligned] [%struct.packed_and_aligned <{ i16 1, i8 2, float 3.000000e+00, i8 undef }>, %struct.packed_and_aligned <{ i16 0, i8 0, float 0.000000e+00, i8 undef }>]
// OGCG-DAG: @_ZZ8functionvE6ba_bf1 = internal constant { i8, i8, i8, i8 } { i8 -1, i8 -86, i8 52, i8 18 }
// OGCG-DAG: @_ZZ8functionvE6ba_bf2 = internal constant { i8, i8, [2 x i8] } { i8 -1, i8 127, [2 x i8] undef }
// OGCG-DAG: @_ZZ8functionvE6ba_bf3 = internal constant %struct.single_byte_bitfield { i8 42 }
// OGCG-DAG: @_ZZ8functionvE5p_bf1 = internal constant { i8, i8, [2 x i8] } { i8 17, i8 3, [2 x i8] undef }
// OGCG-DAG: @_ZZ8functionvE5p_bf2 = internal constant %struct.signed_partial_bitfields { i8 127, [3 x i8] undef }
// OGCG-DAG: @_ZZ8functionvE5p_bf3 = internal constant %struct.mixed_partial_bitfields { i8 125 }

// OGCG-LABEL: define{{.*}} void @_Z8functionv
// OGCG:   ret void
