// RUN: %clang_cc1 -std=c2y -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c2y -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c2y -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

unsigned long vla_with_array_element_type_with_const_size() {
  long size;
  return _Countof(int[5][size]);
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca "" {{.*}} return_slot : !cir.ptr<!u64i>
// CIR: %[[SIZE_ADDR:.*]] = cir.alloca "size" {{.*}} : !cir.ptr<!s64i>
// CIR: %[[CONST_5:.*]] = cir.const #cir.int<5> : !u64i
// CIR: cir.store %[[CONST_5]], %[[RET_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[RET_VAL:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: cir.return %[[RET_VAL]] : !u64i

// LLVM: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// LLVM: ret i64 5

// OGCG: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// OGCG: ret i64 5

unsigned long vla_with_array_element_type_non_const_size() {
  long size;
  return _Countof(int[size][size]);
}

// CIR: %[[REET_ADDR:.*]] = cir.alloca "" {{.*}} return_slot : !cir.ptr<!u64i>
// CIR: %[[SIZE_ADDR:.*]] = cir.alloca "size" {{.*}} : !cir.ptr<!s64i>
// CIR: %[[TMP_SIZE:.*]] = cir.load {{.*}} %[[SIZE_ADDR]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[TMP_SIZE_U64:.*]] = cir.cast integral %[[TMP_SIZE]] : !s64i -> !u64i
// CIR: cir.store %[[TMP_SIZE_U64]], %[[RET_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: cir.return %[[TMP_RET]] : !u64i

// LLVM: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// LLVM: %[[TMP_SIZE:.*]] = load i64, ptr %[[SIZE_ADDR]], align 8
// LLVM: ret i64 %[[TMP_SIZE]]

// OGCG: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[TMP_SIZE:.*]] = load i64, ptr %[[SIZE_ADDR]], align 8
// OGCG: %[[TMP_SIZE_2:.*]] = load i64, ptr %[[SIZE_ADDR]], align 8
// OGCG: ret i64 %[[TMP_SIZE]]
