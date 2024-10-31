// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// CHECK: ![[I128_STRUCT:.+]] = !cir.struct<struct  {!s64i, !s64i}>

// CHECK: @_Z5test1nn(%[[ARG0:.+]]: !s64i loc({{.+}}), %[[ARG1:.+]]: !s64i loc({{.+}}), %[[ARG2:.+]]: !s64i loc({{.+}}), %[[ARG3:.+]]: !s64i loc({{.+}})) -> ![[I128_STRUCT]]
// LLVM: define dso_local { i64, i64 } @_Z5test1nn(i64 %[[#A_LO:]], i64 %[[#A_HI:]], i64 %[[#B_LO:]], i64 %[[#B_HI:]])
__int128 test1(__int128 a, __int128 b) {
  //      CHECK: %[[#SLOT_A:]] = cir.alloca !s128i, !cir.ptr<!s128i>
  // CHECK-NEXT: %[[#SLOT_A2:]] = cir.cast(bitcast, %[[#SLOT_A]] : !cir.ptr<!s128i>), !cir.ptr<![[I128_STRUCT]]>
  // CHECK-NEXT: %[[#SLOT_A_LO:]] = cir.get_member %[[#SLOT_A2]][0] {name = ""} : !cir.ptr<![[I128_STRUCT]]> -> !cir.ptr<!s64i>
  // CHECK-NEXT: cir.store %[[ARG0]], %[[#SLOT_A_LO]] : !s64i, !cir.ptr<!s64i>
  // CHECK-NEXT: %[[#SLOT_A_HI:]] = cir.get_member %[[#SLOT_A2]][1] {name = ""} : !cir.ptr<![[I128_STRUCT]]> -> !cir.ptr<!s64i>
  // CHECK-NEXT: cir.store %arg1, %[[#SLOT_A_HI]] : !s64i, !cir.ptr<!s64i>
  // CHECK-NEXT: %[[#SLOT_B:]] = cir.alloca !s128i, !cir.ptr<!s128i>
  // CHECK-NEXT: %[[#SLOT_B2:]] = cir.cast(bitcast, %[[#SLOT_B]] : !cir.ptr<!s128i>), !cir.ptr<![[I128_STRUCT]]>
  // CHECK-NEXT: %[[#SLOT_B_LO:]] = cir.get_member %[[#SLOT_B2]][0] {name = ""} : !cir.ptr<![[I128_STRUCT]]> -> !cir.ptr<!s64i>
  // CHECK-NEXT: cir.store %arg2, %[[#SLOT_B_LO]] : !s64i, !cir.ptr<!s64i>
  // CHECK-NEXT: %[[#SLOT_B_HI:]] = cir.get_member %[[#SLOT_B2]][1] {name = ""} : !cir.ptr<![[I128_STRUCT]]> -> !cir.ptr<!s64i>
  // CHECK-NEXT: cir.store %arg3, %[[#SLOT_B_HI]] : !s64i, !cir.ptr<!s64i>
  // CHECK-NEXT: %[[#SLOT_RET:]] = cir.alloca !s128i, !cir.ptr<!s128i>, ["__retval"]

  //      LLVM: %[[#A_SLOT:]] = alloca i128, i64 1, align 4
  // LLVM-NEXT: %[[#A_SLOT_LO:]] = getelementptr { i64, i64 }, ptr %[[#A_SLOT]], i32 0, i32 0
  // LLVM-NEXT: store i64 %[[#A_LO]], ptr %[[#A_SLOT_LO]], align 8
  // LLVM-NEXT: %[[#A_SLOT_HI:]] = getelementptr { i64, i64 }, ptr %[[#A_SLOT]], i32 0, i32 1
  // LLVM-NEXT: store i64 %[[#A_HI]], ptr %[[#A_SLOT_HI]], align 8
  // LLVM-NEXT: %[[#B_SLOT:]] = alloca i128, i64 1, align 4
  // LLVM-NEXT: %[[#B_SLOT_LO:]] = getelementptr { i64, i64 }, ptr %[[#B_SLOT]], i32 0, i32 0
  // LLVM-NEXT: store i64 %[[#B_LO]], ptr %[[#B_SLOT_LO]], align 8
  // LLVM-NEXT: %[[#B_SLOT_HI:]] = getelementptr { i64, i64 }, ptr %[[#B_SLOT]], i32 0, i32 1
  // LLVM-NEXT: store i64 %[[#B_HI]], ptr %[[#B_SLOT_HI]], align 8
  // LLVM-NEXT: %[[#RET_SLOT:]] = alloca i128, i64 1, align 16

  return a + b;
  //      CHECK: %[[#A:]] = cir.load %[[#SLOT_A]] : !cir.ptr<!s128i>, !s128i
  // CHECK-NEXT: %[[#B:]] = cir.load %[[#SLOT_B]] : !cir.ptr<!s128i>, !s128i
  // CHECK-NEXT: %[[#SUM:]] = cir.binop(add, %[[#A]], %[[#B]]) nsw : !s128i
  // CHECK-NEXT: cir.store %[[#SUM]], %[[#SLOT_RET]] : !s128i, !cir.ptr<!s128i>

  //      LLVM: %[[#A:]] = load i128, ptr %5, align 16
  // LLVM-NEXT: %[[#B:]] = load i128, ptr %8, align 16
  // LLVM-NEXT: %[[#SUM:]] = add nsw i128 %[[#A]], %[[#B]]
  // LLVM-NEXT: store i128 %[[#SUM]], ptr %[[#RET_SLOT]], align 16

  //      CHECK: %[[#SLOT_RET2:]] = cir.cast(bitcast, %[[#SLOT_RET]] : !cir.ptr<!s128i>), !cir.ptr<![[I128_STRUCT]]>
  // CHECK-NEXT: %[[#RET:]] = cir.load %[[#SLOT_RET2]] : !cir.ptr<![[I128_STRUCT]]>, ![[I128_STRUCT]]
  // CHECK-NEXT: cir.return %[[#RET]] : ![[I128_STRUCT]]

  //      LLVM: %[[#RET:]] = load { i64, i64 }, ptr %[[#RET_SLOT]], align 8
  // LLVM-NEXT: ret { i64, i64 } %[[#RET]]
}
