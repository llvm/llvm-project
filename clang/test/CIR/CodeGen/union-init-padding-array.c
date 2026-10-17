// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

union Wide { char text[12]; unsigned d[3]; };
union Word { char text[4]; unsigned d; };
union Byte { char c; unsigned d; };
union Str { int id; char *str; };
union Odd { char c[5]; int i[2]; };
union Ptr { int *p; long double ld; };

union Wide cpuid[1] = { { "GenuineIntel" } };
// CIR: cir.global external @cpuid = #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<71> : !s8i, #cir.int<101> : !s8i, #cir.int<110> : !s8i, #cir.int<117> : !s8i, #cir.int<105> : !s8i, #cir.int<110> : !s8i, #cir.int<101> : !s8i, #cir.int<73> : !s8i, #cir.int<110> : !s8i, #cir.int<116> : !s8i, #cir.int<101> : !s8i, #cir.int<108> : !s8i]> : !cir.array<!s8i x 12>}> : !rec_Wide]> : !cir.array<!rec_Wide x 1> 
// LLVM: @cpuid = global [1 x { [12 x i8] }] [{ [12 x i8] } { [12 x i8] c"GenuineIntel" }]

union Word word[1] = { { "abcd" } };
// CIR: cir.global external @word = #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i, #cir.int<100> : !s8i]> : !cir.array<!s8i x 4>}> : !rec_Word]> : !cir.array<!rec_Word x 1> 
// LLVM: @word = global [1 x { [4 x i8] }] [{ [4 x i8] } { [4 x i8] c"abcd" }]

union Byte byte[1] = { { 'a' } };
// CIR: cir.global external @byte = #cir.const_array<[#cir.const_record<{#cir.int<97> : !s8i}> : !rec_Byte]> : !cir.array<!rec_Byte x 1> 
// LLVM: @byte = global [1 x { i8, [3 x i8] }] [{ i8, [3 x i8] } { i8 97, [3 x i8] {{.*}} }]

union Odd odd[2] = { { .c = "ab" }, { .c = "cd" } };
// CIR: cir.global external @odd = #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i], trailing_zeros> : !cir.array<!s8i x 5>}> : !rec_Odd, #cir.const_record<{#cir.const_array<[#cir.int<99> : !s8i, #cir.int<100> : !s8i], trailing_zeros> : !cir.array<!s8i x 5>}> : !rec_Odd]> : !cir.array<!rec_Odd x 2> 
// LLVM: @odd = global [2 x { [5 x i8], [3 x i8] }] [{ [5 x i8], [3 x i8] } { [5 x i8] c"ab\00\00\00", [3 x i8] {{.*}} }, { [5 x i8], [3 x i8] } { [5 x i8] c"cd\00\00\00", [3 x i8] {{.*}} }]

union Wide same[2] = { { .text = "abc" }, { .text = "xyz" } };
// CIR: cir.global external @same = #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide, #cir.const_record<{#cir.const_array<[#cir.int<120> : !s8i, #cir.int<121> : !s8i, #cir.int<122> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide]> : !cir.array<!rec_Wide x 2> 
// LLVM: @same = global [2 x { [12 x i8] }] [{ [12 x i8] } { [12 x i8] c"abc\00\00\00\00\00\00\00\00\00" }, { [12 x i8] } { [12 x i8] c"xyz\00\00\00\00\00\00\00\00\00" }]

union Wide mixed[2] = { { .text = "abc" }, { .d = { 1, 2, 3 } } };
// CIR: cir.global external @mixed = #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide, #cir.const_record<{#cir.const_array<[#cir.int<1> : !u32i, #cir.int<2> : !u32i, #cir.int<3> : !u32i]> : !cir.array<!u32i x 3>}> : !rec_Wide]> : !cir.array<!rec_Wide x 2> 
// LLVM: @mixed = global <{ { [12 x i8] }, %union.Wide }> <{ { [12 x i8] } { [12 x i8] c"abc\00\00\00\00\00\00\00\00\00" }, %union.Wide { [3 x i32] [i32 1, i32 2, i32 3] } }>

union Wide tail[2] = { { .text = "abc" } };
// CIR: cir.global external @tail = #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide], trailing_zeros> : !cir.array<!rec_Wide x 2> 
// LLVM: @tail = global <{ { [12 x i8] }, %union.Wide }> <{ { [12 x i8] } { [12 x i8] c"abc\00\00\00\00\00\00\00\00\00" }, %union.Wide zeroinitializer }>

union Wide grid[2][2] = { { { .text = "a" }, { .text = "b" } },
                          { { .text = "c" }, { .text = "d" } } };
// CIR: cir.global external @grid = #cir.const_array<[#cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide, #cir.const_record<{#cir.const_array<[#cir.int<98> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide]> : !cir.array<!rec_Wide x 2>, #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<99> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide, #cir.const_record<{#cir.const_array<[#cir.int<100> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide]> : !cir.array<!rec_Wide x 2>]> : !cir.array<!cir.array<!rec_Wide x 2> x 2> 
// LLVM: @grid = global [2 x [2 x { [12 x i8] }]] {{\[\[}}2 x { [12 x i8] }] [{ [12 x i8] } { [12 x i8] c"a\00\00\00\00\00\00\00\00\00\00\00" }, { [12 x i8] } { [12 x i8] c"b\00\00\00\00\00\00\00\00\00\00\00" }], [2 x { [12 x i8] }] [{ [12 x i8] } { [12 x i8] c"c\00\00\00\00\00\00\00\00\00\00\00" }, { [12 x i8] } { [12 x i8] c"d\00\00\00\00\00\00\00\00\00\00\00" }]]

struct PadElem { int info; union Str u; };
struct PadElem pad_elems[2] = { { 1, { 2 } }, { 3, { 4 } } };
// CIR: cir.global external @pad_elems = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i, #cir.const_record<{#cir.int<2> : !s32i}> : !rec_Str}> : !rec_PadElem, #cir.const_record<{#cir.int<3> : !s32i, #cir.const_record<{#cir.int<4> : !s32i}> : !rec_Str}> : !rec_PadElem]> : !cir.array<!rec_PadElem x 2> 
// LLVM: @pad_elems = global [2 x { i32, [4 x i8], { i32, [4 x i8] } }] [{ i32, [4 x i8], { i32, [4 x i8] } } { i32 1, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 2, [4 x i8] {{.*}} } }, { i32, [4 x i8], { i32, [4 x i8] } } { i32 3, [4 x i8] zeroinitializer, { i32, [4 x i8] } { i32 4, [4 x i8] {{.*}} } }]

struct PadHolder { int a; union Str arr[2]; };
struct PadHolder pad_holder = { 1, { { 2 }, { 3 } } };
// CIR: cir.global external @pad_holder = #cir.const_record<{#cir.int<1> : !s32i, #cir.const_array<[#cir.const_record<{#cir.int<2> : !s32i}> : !rec_Str, #cir.const_record<{#cir.int<3> : !s32i}> : !rec_Str]> : !cir.array<!rec_Str x 2>}> : !rec_PadHolder 
// LLVM: @pad_holder = global { i32, [4 x i8], [2 x { i32, [4 x i8] }] } { i32 1, [4 x i8] zeroinitializer, [2 x { i32, [4 x i8] }] [{ i32, [4 x i8] } { i32 2, [4 x i8] {{.*}} }, { i32, [4 x i8] } { i32 3, [4 x i8] {{.*}} }] }

struct Holder { int a; union Wide arr[2]; int b; };
struct Holder holder = { 1, { { .text = "xy" }, { .text = "zw" } }, 2 };
// CIR: cir.global external @holder = #cir.const_record<{#cir.int<1> : !s32i, #cir.const_array<[#cir.const_record<{#cir.const_array<[#cir.int<120> : !s8i, #cir.int<121> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide, #cir.const_record<{#cir.const_array<[#cir.int<122> : !s8i, #cir.int<119> : !s8i], trailing_zeros> : !cir.array<!s8i x 12>}> : !rec_Wide]> : !cir.array<!rec_Wide x 2>, #cir.int<2> : !s32i}> : !rec_Holder 
// LLVM: @holder = global { i32, [2 x { [12 x i8] }], i32 } { i32 1, [2 x { [12 x i8] }] [{ [12 x i8] } { [12 x i8] c"xy\00\00\00\00\00\00\00\00\00\00" }, { [12 x i8] } { [12 x i8] c"zw\00\00\00\00\00\00\00\00\00\00" }], i32 2 }

extern int ext[4];
union Ptr relocs[2] = { { .p = &ext[1] }, { .p = &ext[2] } };
// CIR: cir.global external @relocs = #cir.const_array<[#cir.const_record<{#cir.global_view<@ext, [1 : i32]> : !cir.ptr<!s32i>}> : !rec_Ptr, #cir.const_record<{#cir.global_view<@ext, [2 : i32]> : !cir.ptr<!s32i>}> : !rec_Ptr]> : !cir.array<!rec_Ptr x 2>
// LLVM: @relocs = global [2 x { ptr, [8 x i8] }] [{ ptr, [8 x i8] } { ptr getelementptr {{.*}}(i8, ptr @ext, i64 4), [8 x i8] {{.*}} }, { ptr, [8 x i8] } { ptr getelementptr {{.*}}(i8, ptr @ext, i64 8), [8 x i8] {{.*}} }]

union U { int id; char *str; };
struct FamU { int n; union U fam[]; };
struct FamU flex_array_mem = { 3, { { 1 }, { 2 } } };
// CIR: cir.global external @flex_array_mem = #cir.const_record<{#cir.int<3> : !s32i, #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i}> : !rec_U, #cir.const_record<{#cir.int<2> : !s32i}> : !rec_U]> : !cir.array<!rec_U x 2>}> : !rec_FamU
// LLVM: @flex_array_mem = global { i32, [4 x i8], [2 x { i32, [4 x i8] }] } { i32 3, [4 x i8] zeroinitializer, [2 x { i32, [4 x i8] }] [{ i32, [4 x i8] } { i32 1, [4 x i8] zeroinitializer }, { i32, [4 x i8] } { i32 2, [4 x i8] zeroinitializer }] }

union U2 { char c; unsigned long long ull; };
struct FamU2 { int n; union U2 fam[]; };

struct FamU2 flex_array_mem2 = { 3, { { 1 }, { 'c' } } };
// CIR: cir.global external @flex_array_mem2 = #cir.const_record<{#cir.int<3> : !s32i, #cir.const_array<[#cir.const_record<{#cir.int<1> : !s8i}> : !rec_U2, #cir.const_record<{#cir.int<99> : !s8i}> : !rec_U2]> : !cir.array<!rec_U2 x 2>}> : !rec_FamU2
// LLVM: @flex_array_mem2 = global { i32, [4 x i8], [2 x { i8, [7 x i8] }] } { i32 3, [4 x i8] zeroinitializer, [2 x { i8, [7 x i8] }] [{ i8, [7 x i8] } { i8 1, [7 x i8] zeroinitializer }, { i8, [7 x i8] } { i8 99, [7 x i8] zeroinitializer }] }
//
struct FamU2 flex_array_mem3 = { 3, { { 'c' }, { 1 } } };
// CIR: cir.global external @flex_array_mem3 = #cir.const_record<{#cir.int<3> : !s32i, #cir.const_array<[#cir.const_record<{#cir.int<99> : !s8i}> : !rec_U2, #cir.const_record<{#cir.int<1> : !s8i}> : !rec_U2]> : !cir.array<!rec_U2 x 2>}> : !rec_FamU2
// LLVM: @flex_array_mem3 = global { i32, [4 x i8], [2 x { i8, [7 x i8] }] } { i32 3, [4 x i8] zeroinitializer, [2 x { i8, [7 x i8] }] [{ i8, [7 x i8] } { i8 99, [7 x i8] zeroinitializer }, { i8, [7 x i8] } { i8 1, [7 x i8] zeroinitializer }] }

