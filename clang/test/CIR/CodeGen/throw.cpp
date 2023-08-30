// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

double d(int a, int b) {
   if (b == 0)
      throw "Division by zero condition!";
   return (a/b);
}

//      CHECK: cir.if %10 {
// CHECK-NEXT:   %11 = cir.alloc_exception(!cir.ptr<!s8i>) -> <!cir.ptr<!s8i>>
// CHECK-NEXT:   %12 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 28>>
// CHECK-NEXT:   %13 = cir.cast(array_to_ptrdecay, %12 : !cir.ptr<!cir.array<!s8i x 28>>), !cir.ptr<!s8i>
// CHECK-NEXT:   cir.store %13, %11 : !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>
// CHECK-NEXT:   cir.throw(%11 : !cir.ptr<!cir.ptr<!s8i>>, @_ZTIPKc)
// CHECK-NEXT: }