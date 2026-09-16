// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG

// FIXME(cir): Once we implement reassociate in CIR, we should check this again
// to make sure we actually emit the fence.

void complex_test() {
  double _Complex a;
  double _Complex r = __arithmetic_fence(a); }

// CIR-LABEL: cir.func {{.*}}@complex_test()
// CIR-NEXT:  %[[A_ALLOC:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.complex<!cir.double>>
// CIR-NEXT:  %[[R_ALLOC:.*]] = cir.alloca "r" {{.*}} init : !cir.ptr<!cir.complex<!cir.double>>
// CIR-NEXT:  %[[LOAD:.*]] = cir.load {{.*}}%[[A_ALLOC]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR-NEXT:  cir.store {{.*}}%[[LOAD]], %[[R_ALLOC]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CIR-NEXT:  cir.return

// Differences are just how we represent Complex in CIR.
// LLVM-LABEL: define {{.*}}void @complex_test()
// OGCG-NEXT:   entry:
// LLVM-NEXT:      %[[A_ALLOC:.*]] = alloca { double, double }
// LLVM-NEXT:      %[[R_ALLOC:.*]] = alloca { double, double }
// LLVMCIR-NEXT:   %[[LOAD:.*]] = load { double, double }, ptr %[[A_ALLOC]]
// LLVMCIR-NEXT:   store { double, double } %[[LOAD]], ptr %[[R_ALLOC]]
// OGCG-NEXT:      %[[A_REALP:.*]] = getelementptr inbounds nuw { double, double }, ptr %a, i32 0, i32 0
// OGCG-NEXT:      %[[A_REAL:.*]] = load double, ptr %[[A_REALP]]
// OGCG-NEXT:      %[[A_IMAGP:.*]] = getelementptr inbounds nuw { double, double }, ptr %a, i32 0, i32 1
// OGCG-NEXT:      %[[A_IMAG:.*]] = load double, ptr %[[A_IMAGP]]
// OGCG-NEXT:      %[[R_REALP:.*]] = getelementptr inbounds nuw { double, double }, ptr %r, i32 0, i32 0
// OGCG-NEXT:      %[[R_IMAGP:.*]] = getelementptr inbounds nuw { double, double }, ptr %r, i32 0, i32 1
// OGCG-NEXT:      store double %[[A_REAL]], ptr %[[R_REALP]]
// OGCG-NEXT:      store double %[[A_IMAG]], ptr %[[R_IMAGP]]
// LLVMCIR-NEXT:   ret void

// Scalar floating-point operand.
void scalar_test() {
  double a;
  double r = __arithmetic_fence(a);
}

// CIR-LABEL: cir.func {{.*}}@scalar_test()
// CIR-NEXT:  %[[A_ALLOC:.*]] = cir.alloca "a" {{.*}}: !cir.ptr<!cir.double>
// CIR-NEXT:  %[[R_ALLOC:.*]] = cir.alloca "r" {{.*}}init : !cir.ptr<!cir.double>
// CIR-NEXT:  %[[LOAD:.*]] = cir.load {{.*}} %[[A_ALLOC]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT:  cir.store {{.*}}%[[LOAD]], %[[R_ALLOC]] : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT:  cir.return

// LLVM-LABEL: define {{.*}}void @scalar_test()
// OGCG-NEXT:   entry:
// LLVM-NEXT:   %[[A_ALLOC:.*]] = alloca double
// LLVM-NEXT:   %[[R_ALLOC:.*]] = alloca double
// LLVM-NEXT:   %[[LOAD:.*]] = load double, ptr %[[A_ALLOC]]
// LLVM-NEXT:   store double %[[LOAD]], ptr %[[R_ALLOC]]
// LLVM-NEXT:   ret void

// Vector operand: scalar-evaluated, but must not be routed to the target path.
typedef float float4 __attribute__((ext_vector_type(4)));
void vec_test() {
  float4 a;
  float4 r = __arithmetic_fence(a);
}

// CIR-LABEL: cir.func no_inline no_proto dso_local @vec_test()
// CIR-NEXT:   %[[A_ALLOC:.*]] = cir.alloca "a" {{.*}}: !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR-NEXT:   %[[R_ALLOC:.*]] = cir.alloca "r" {{.*}}init : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR-NEXT:   %[[LOAD:.*]] = cir.load {{.*}}%[[A_ALLOC]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR-NEXT:   cir.store {{.*}}%[[LOAD]], %[[R_ALLOC]] : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR-NEXT:   cir.return

// LLVM-LABEL: define {{.*}}void @vec_test()
// OGCG-NEXT:   entry:
// LLVM-NEXT:   %[[A_ALLOC:.*]] = alloca <4 x float>
// LLVM-NEXT:   %[[R_ALLOC:.*]] = alloca <4 x float>
// LLVM-NEXT:   %[[LOAD:.*]] = load <4 x float>, ptr %[[A_ALLOC]]
// LLVM-NEXT:   store <4 x float> %[[LOAD]], ptr %[[R_ALLOC]]
// LLVM-NEXT:   ret void
