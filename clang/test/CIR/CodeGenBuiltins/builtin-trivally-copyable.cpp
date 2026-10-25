// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG

bool g;
void store_trivially_copyable_result() {
  g = __is_trivially_copyable(int);
}

// CIR: cir.func {{.*}} @_Z31store_trivially_copyable_resultv()
// CIR:   %[[TRUE:.*]] = cir.const #true
// CIR:   %[[G_PTR:.*]] = cir.get_global @g : !cir.ptr<!cir.bool>
// CIR:   cir.store{{.*}} %[[TRUE]], %[[G_PTR:.*]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: define{{.*}} void @_Z31store_trivially_copyable_resultv()
// LLVM:   store i8 1, ptr @g

int test_trivially_copyable_as_bool() {
  if (!__is_trivially_copyable(int))
    return -1;
  return 0;
}

// CIR: cir.func {{.*}} @_Z31test_trivially_copyable_as_boolv()
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[RETVAL:.*]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL:.*]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RET:.*]] : !s32i

// LLVMCIR: define{{.*}} i32 @_Z31test_trivially_copyable_as_boolv()
// LLVMCIR:   store i32 0, ptr %[[RETVAL:.*]]
// LLVMCIR:   %[[RET:.*]] = load i32, ptr %[[RETVAL:.*]]
// LLVMCIR:   ret i32 %[[RET:.*]]

// OGCG: define{{.*}} i32 @_Z31test_trivially_copyable_as_boolv()
// OGCG:   ret i32 0
