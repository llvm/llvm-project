// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t2.cir 2>&1 | FileCheck -check-prefix=AFTER %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void foo() {
  char s1[] = "Hello";
}
// AFTER-DAG:  cir.global "private"  constant cir_private @__const._Z3foov.s1 = #cir.const_array<"Hello\00" : !cir.array<!s8i x 6>> : !cir.array<!s8i x 6>
// AFTER: @_Z3foov
// AFTER:    %[[S1:.*]] = cir.alloca !cir.array<!s8i x 6>, !cir.ptr<!cir.array<!s8i x 6>>, ["s1"]
// AFTER:    %[[HELLO:.*]] = cir.get_global @__const._Z3foov.s1 : !cir.ptr<!cir.array<!s8i x 6>>
// AFTER:    cir.copy %[[HELLO]] to %[[S1]] : !cir.ptr<!cir.array<!s8i x 6>>
// AFTER:    cir.return
// AFTER:  }

// LLVM: @__const._Z3foov.s1 = private constant [6 x i8] c"Hello\00"
// LLVM: @_Z3foov()
// LLVM:   %[[S1:.*]] = alloca [6 x i8], i64 1, align 1
// FIXME: LLVM OG uses @llvm.memcpy.p0.p0.i64
// LLVM:   call void @llvm.memcpy.p0.p0.i32(ptr %[[S1]], ptr @__const._Z3foov.s1, i32 6, i1 false)
// LLVM:   ret void