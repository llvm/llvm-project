// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// An array global ending in 8 or more zeros lowers, as in classic codegen, to
// a packed struct of its explicit elements and one zero-initialized array, so
// the zeros are never spelled out. Spelling out the 2 GiB arrays here would
// not finish.

unsigned char huge[1024 * 1024 * 1024 * 2u] = {1};
// CIR: cir.global external @huge = #cir.const_array<[#cir.int<1> : !u8i], trailing_zeros> : !cir.array<!u8i x 2147483648>
// LLVM: @huge = global <{ i8, [2147483647 x i8] }> <{ i8 1, [2147483647 x i8] zeroinitializer }>

unsigned char cube[1024][1024][1024] = {{{1}}};
// LLVM: @cube = global <{ <{ <{ i8, [1023 x i8] }>, [1023 x <{ i8, [1023 x i8] }>] }>, [1023 x <{ <{ i8, [1023 x i8] }>, [1023 x <{ i8, [1023 x i8] }>] }>] }> <{ <{ <{ i8, [1023 x i8] }>, [1023 x <{ i8, [1023 x i8] }>] }> <{ <{ i8, [1023 x i8] }> <{ i8 1, [1023 x i8] zeroinitializer }>, [1023 x <{ i8, [1023 x i8] }>] zeroinitializer }>, [1023 x <{ <{ i8, [1023 x i8] }>, [1023 x <{ i8, [1023 x i8] }>] }>] zeroinitializer }>

// Address target for the pointer arrays below.
int few[4] = {1};

int inner[2][10] = {{1}, {2, 3}};
// LLVM: @inner = global <{ <{ i32, [9 x i32] }>, <{ i32, i32, [8 x i32] }> }> <{ <{ i32, [9 x i32] }> <{ i32 1, [9 x i32] zeroinitializer }>, <{ i32, i32, [8 x i32] }> <{ i32 2, i32 3, [8 x i32] zeroinitializer }> }>

int *ptrs[10] = {&few[0], &few[1]};
// LLVM: @ptrs = global <{ ptr, ptr, [8 x ptr] }> <{ ptr @few, ptr getelementptr {{.*}}(i8, ptr @few, i64 4), [8 x ptr] zeroinitializer }>

struct Rec { int a; char c[12]; } recs[2] = {{1, {'x'}}, {2}};
// LLVM: @recs = global <{ { i32, <{ i8, [11 x i8] }> }, %struct.Rec }> <{ { i32, <{ i8, [11 x i8] }> } { i32 1, <{ i8, [11 x i8] }> <{ i8 120, [11 x i8] zeroinitializer }> }, %struct.Rec { i32 2, [12 x i8] zeroinitializer } }>

// An address with an offset needs the insertvalue initializer region, which
// builds the same packed struct.
int *many_ptrs[16] = {&few[1], &few[1], &few[1], &few[1],
                      &few[1], &few[1], &few[1], &few[1]};
// LLVM: @many_ptrs = global <{ [8 x ptr], [8 x ptr] }> <{ [8 x ptr] [ptr getelementptr {{.*}}(i8, ptr @few, i64 4), ptr getelementptr {{.*}}(i8, ptr @few, i64 4), ptr getelementptr {{.*}}(i8, ptr @few, i64 4), ptr getelementptr {{.*}}(i8, ptr @few, i64 4), ptr getelementptr {{.*}}(i8, ptr @few, i64 4), ptr getelementptr {{.*}}(i8, ptr @few, i64 4), ptr getelementptr {{.*}}(i8, ptr @few, i64 4), ptr getelementptr {{.*}}(i8, ptr @few, i64 4)], [8 x ptr] zeroinitializer }>

int *ptr_grid[2][10] = {{&few[1]}, {&few[2]}};
// LLVM: @ptr_grid = global [2 x <{ ptr, [9 x ptr] }>] [<{ ptr, [9 x ptr] }> <{ ptr getelementptr {{.*}}(i8, ptr @few, i64 4), [9 x ptr] zeroinitializer }>, <{ ptr, [9 x ptr] }> <{ ptr getelementptr {{.*}}(i8, ptr @few, i64 8), [9 x ptr] zeroinitializer }>]
