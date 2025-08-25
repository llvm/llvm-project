// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s


char const *array[] {
    "my", "hands", "are", "typing", "words"
};

// CIR: cir.global "private" constant cir_private dso_local @"[[STR:.+]]" = #cir.const_array<"my\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>
// CIR: cir.global "private" constant cir_private dso_local @"[[STR1:.+]]" = #cir.const_array<"hands\00" : !cir.array<!s8i x 6>> : !cir.array<!s8i x 6>
// CIR: cir.global "private" constant cir_private dso_local @"[[STR2:.+]]" = #cir.const_array<"are\00" : !cir.array<!s8i x 4>> : !cir.array<!s8i x 4>
// CIR: cir.global "private" constant cir_private dso_local @"[[STR3:.+]]" = #cir.const_array<"typing\00" : !cir.array<!s8i x 7>> : !cir.array<!s8i x 7>
// CIR: cir.global "private" constant cir_private dso_local @"[[STR4:.+]]" = #cir.const_array<"words\00" : !cir.array<!s8i x 6>> : !cir.array<!s8i x 6>
// CIR: cir.global external @array = #cir.const_array<[#cir.global_view<@"[[STR]]"> : !cir.ptr<!s8i>, #cir.global_view<@"[[STR1]]"> : !cir.ptr<!s8i>, #cir.global_view<@"[[STR2]]"> : !cir.ptr<!s8i>, #cir.global_view<@"[[STR3]]"> : !cir.ptr<!s8i>, #cir.global_view<@"[[STR4]]"> : !cir.ptr<!s8i>]> : !cir.array<!cir.ptr<!s8i> x 5>

// LLVM: @[[STR:.+]] = private constant [3 x i8] c"my\00"
// LLVM: @[[STR1:.+]] = private constant [6 x i8] c"hands\00"
// LLVM: @[[STR2:.+]] = private constant [4 x i8] c"are\00"
// LLVM: @[[STR3:.+]] = private constant [7 x i8] c"typing\00"
// LLVM: @[[STR4:.+]] = private constant [6 x i8] c"words\00"
// LLVM: @array = global [5 x ptr] [ptr @[[STR]], ptr @[[STR1]], ptr @[[STR2]], ptr @[[STR3]], ptr @[[STR4]]]

// OGCG: @[[STR:.+]] = private unnamed_addr constant [3 x i8] c"my\00"
// OGCG: @[[STR1:.+]] = private unnamed_addr constant [6 x i8] c"hands\00"
// OGCG: @[[STR2:.+]] = private unnamed_addr constant [4 x i8] c"are\00"
// OGCG: @[[STR3:.+]] = private unnamed_addr constant [7 x i8] c"typing\00"
// OGCG: @[[STR4:.+]] = private unnamed_addr constant [6 x i8] c"words\00"
// OGCG: @array = global [5 x ptr] [ptr @[[STR]], ptr @[[STR1]], ptr @[[STR2]], ptr @[[STR3]], ptr @[[STR4]]]

// CIR: cir.global "private" constant cir_private dso_local @[[STR5_GLOBAL:.*]] = #cir.const_array<"abcd\00" : !cir.array<!s8i x 5>> : !cir.array<!s8i x 5>

// LLVM: @[[STR5_GLOBAL:.*]] = private constant [5 x i8] c"abcd\00"

// OGCG: @[[STR5_GLOBAL:.*]] = private unnamed_addr constant [5 x i8] c"abcd\00"

decltype(auto) returns_literal() {
    return "abcd";
}

// CIR: cir.func{{.*}} @_Z15returns_literalv() -> !cir.ptr<!cir.array<!s8i x 5>>
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !cir.ptr<!cir.array<!s8i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s8i x 5>>>, ["__retval"]
// CIR:   %[[STR_ADDR:.*]] = cir.get_global @[[STR5_GLOBAL]] : !cir.ptr<!cir.array<!s8i x 5>>
// CIR:   cir.store{{.*}} %[[STR_ADDR]], %[[RET_ADDR]]
// CIR:   %[[RET:.*]] = cir.load %[[RET_ADDR]]
// CIR:   cir.return %[[RET]]
