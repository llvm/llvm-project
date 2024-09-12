// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

double d(int a, int b) {
   if (b == 0)
      throw "Division by zero condition!";
   return (a/b);
}

//      CIR: cir.if
// CIR-NEXT:   %[[ADDR:.*]] = cir.alloc.exception 8
// CIR-NEXT:   %[[STR:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 28>>
// CIR-NEXT:   %[[STR_ADD:.*]] = cir.cast(array_to_ptrdecay, %[[STR]] : !cir.ptr<!cir.array<!s8i x 28>>), !cir.ptr<!s8i>
// CIR-NEXT:   cir.store %[[STR_ADD]], %[[ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR-NEXT:   cir.throw %[[ADDR]] : !cir.ptr<!cir.ptr<!s8i>>, @_ZTIPKc
// CIR-NEXT:   cir.unreachable
// CIR-NEXT: }

// LLVM: %[[ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 8)
// LLVM: store ptr @.str, ptr %[[ADDR]], align 8
// LLVM: call void @__cxa_throw(ptr %[[ADDR]], ptr @_ZTIPKc, ptr null)
// LLVM: unreachable