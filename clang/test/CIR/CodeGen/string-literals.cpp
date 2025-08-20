// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// CIR: cir.global "private" constant cir_private dso_local @[[STR1_GLOBAL:.*]] = #cir.const_array<"abcd\00" : !cir.array<!s8i x 5>> : !cir.array<!s8i x 5>

// LLVM: @[[STR1_GLOBAL:.*]] = private constant [5 x i8] c"abcd\00"

// OGCG: @[[STR1_GLOBAL:.*]] = private unnamed_addr constant [5 x i8] c"abcd\00"

decltype(auto) returns_literal() {
    return "abcd";
}

// CIR: cir.func{{.*}} @_Z15returns_literalv() -> !cir.ptr<!cir.array<!s8i x 5>>
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !cir.ptr<!cir.array<!s8i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s8i x 5>>>, ["__retval"]
// CIR:   %[[STR_ADDR:.*]] = cir.get_global @[[STR1_GLOBAL]] : !cir.ptr<!cir.array<!s8i x 5>>
// CIR:   cir.store{{.*}} %[[STR_ADDR]], %[[RET_ADDR]]
// CIR:   %[[RET:.*]] = cir.load %[[RET_ADDR]]
// CIR:   cir.return %[[RET]]
