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

extern "C" char* test_memchr(const char arg[32]) {
  return __builtin_char_memchr(arg, 123, 32);

  // CIR-LABEL: test_memchr
  // CIR: [[PATTERN:%.*]] = cir.const #cir.int<123> : !s32i 
  // CIR: [[LEN:%.*]] = cir.const #cir.int<32> : !s32i 
  // CIR: [[LEN_U64:%.*]] = cir.cast(integral, [[LEN]] : !s32i), !u64i 
  // CIR: {{%.*}} = cir.libc.memchr({{%.*}}, [[PATTERN]], [[LEN_U64]])

  // LLVM: {{.*}}@test_memchr(ptr{{.*}}[[ARG:%.*]]) 
  // LLVM: [[TMP0:%.*]] = alloca ptr, i64 1, align 8
  // LLVM: store ptr [[ARG]], ptr [[TMP0]], align 8
  // LLVM: [[SRC:%.*]] = load ptr, ptr [[TMP0]], align 8
  // LLVM: [[RES:%.*]] = call ptr @memchr(ptr [[SRC]], i32 123, i64 32)
  // LLVM: store ptr [[RES]], ptr [[RET_P:%.*]], align 8
  // LLVM: [[RET:%.*]] = load ptr, ptr [[RET_P]], align 8
  // LLVM: ret ptr [[RET]]
}

extern "C" void *test_return_address(void) {
  return __builtin_return_address(1);

  // CIR-LABEL: test_return_address
  // [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // {{%.*}} = cir.return_address([[ARG]])

  // LLVM-LABEL: @test_return_address
  // LLVM: {{%.*}} = call ptr @llvm.returnaddress(i32 1)
}
