// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int g;
void store_types_compatible_result() {
  g = __builtin_types_compatible_p(int, const int);
}

// CIR: cir.func {{.*}} @store_types_compatible_result()
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   %[[G_PTR:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %[[ONE]], %[[G_PTR:.*]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @store_types_compatible_result()
// LLVM:   store i32 1, ptr @g

// OGCG: define{{.*}} void @store_types_compatible_result()
// OGCG:   store i32 1, ptr @g

int test_convert_bool_to_int() {
  if (!__builtin_types_compatible_p(int, const int))
    return -1;
  return 0;
}

// CIR: cir.func {{.*}} @test_convert_bool_to_int()
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   %[[BOOL:.*]] = cir.cast int_to_bool %[[ONE]] : !s32i -> !cir.bool
// CIR:   %[[NOT:.*]] = cir.unary(not, %[[BOOL]]) : !cir.bool, !cir.bool
// CIR:   cir.if %[[NOT]] {
// CIR:     %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s32i
// CIR:     cir.store %[[NEG_ONE]], %[[RETVAL:.*]]
// CIR:     %[[RET:.*]] = cir.load %[[RETVAL:.*]] : !cir.ptr<!s32i>, !s32i
// CIR:     cir.return %[[RET:.*]] : !s32i
// CIR:   }
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store %[[ZERO]], %[[RETVAL:.*]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL:.*]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RET:.*]] : !s32i

// LLVM: define{{.*}} i32 @test_convert_bool_to_int()
// LLVM:   br i1 false, label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
// LLVM: [[IF_THEN]]:
// LLVM:   store i32 -1, ptr %[[RETVAL:.*]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL:.*]]
// LLVM:   ret i32 %[[RET:.*]]
// LLVM: [[IF_ELSE]]:
// LLVM:   br label %[[IF_END:.*]]
// LLVM: [[IF_END]]:
// LLVM:   store i32 0, ptr %[[RETVAL:.*]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL:.*]]
// LLVM:   ret i32 %[[RET:.*]]

// OGCG: define{{.*}} i32 @test_convert_bool_to_int()
// OGCG:   ret i32 0
