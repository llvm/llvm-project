// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -emit-cir %s -o %t.cir  
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:  -emit-llvm -fno-clangir-call-conv-lowering -o - %s \
// RUN:  | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll 
// RUN: FileCheck  --check-prefix=LLVM --input-file=%t.ll %s

// This test file is a collection of test cases for all target-independent
// builtins that are related to memory operations.

int s;

int *test_addressof() {
  return __builtin_addressof(s);
  
  // CIR-LABEL: test_addressof
  // CIR: [[ADDR:%.*]] = cir.get_global @s : !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  // LLVM-LABEL: test_addressof
  // LLVM: store ptr @s, ptr [[ADDR:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[ADDR]], align 8
  // LLVM: ret ptr [[RES]]
}

namespace std { template<typename T> T *addressof(T &); }
int *test_std_addressof() {
  return std::addressof(s);
  
  // CIR-LABEL: test_std_addressof
  // CIR: [[ADDR:%.*]] = cir.get_global @s : !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  // LLVM-LABEL: test_std_addressof
  // LLVM: store ptr @s, ptr [[ADDR:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[ADDR]], align 8
  // LLVM: ret ptr [[RES]]
}

namespace std { template<typename T> T *__addressof(T &); }
int *test_std_addressof2() {
  return std::__addressof(s);
  
  // CIR-LABEL: test_std_addressof2
  // CIR: [[ADDR:%.*]] = cir.get_global @s : !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  /// LLVM-LABEL: test_std_addressof2
  // LLVM: store ptr @s, ptr [[ADDR:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[ADDR]], align 8
  // LLVM: ret ptr [[RES]]
}
