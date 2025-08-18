// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

double _Complex cd;
float _Complex cf;
int _Complex ci;
short _Complex cs;
double sd;
int si;
bool b;

void scalar_to_complex() {
  cd = sd;
  ci = si;
  cd = si;
  ci = sd;
}

// CIR: %[[FP_TO_COMPLEX:.*]] = cir.cast(float_to_complex, %{{.*}} : !cir.double), !cir.complex<!cir.double>

//      LLVM: %[[REAL:.*]] = load double, ptr {{.*}}, align 8
// LLVM-NEXT: %[[TMP:.*]] = insertvalue { double, double } undef, double %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { double, double } %[[TMP]], double 0.000000e+00, 1
// LLVM-NEXT: store { double, double } %[[COMPLEX]], ptr {{.*}}, align 8

// OGCG: %[[REAL:.*]] = load double, ptr {{.*}}, align 8
// OGCG: store double %[[REAL]], ptr {{.*}}, align 8
// OGCG: store double 0.000000e+00, ptr getelementptr inbounds nuw ({ double, double }, ptr @cd, i32 0, i32 1), align 8

// CIR: %[[INT_TO_COMPLEX:.*]] = cir.cast(int_to_complex, %{{.*}} : !s32i), !cir.complex<!s32i>

//      LLVM: %[[REAL:.*]] = load i32, ptr {{.*}}, align 4
// LLVM-NEXT: %[[TMP:.*]] = insertvalue { i32, i32 } undef, i32 %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { i32, i32 } %[[TMP]], i32 0, 1
// LLVM-NEXT: store { i32, i32 } %[[COMPLEX]], ptr {{.*}}, align 4

// OGCG:  %[[REAL:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: store i32 %[[REAL]], ptr {{.*}}, align 4
// OGCG: store i32 0, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @ci, i32 0, i32 1), align 4

// CIR: %[[INT_TO_FP:.*]] = cir.cast(int_to_float, %{{.*}} : !s32i), !cir.double
// CIR: %[[FP_TO_COMPLEX:.*]] = cir.cast(float_to_complex, %[[INT_TO_FP]] : !cir.double), !cir.complex<!cir.double>

//      LLVM: %[[TMP:.*]] = load i32, ptr {{.*}}, align 4
// LLVM-NEXT: %[[REAL:.*]] = sitofp i32 %[[TMP]] to double
// LLVM-NEXT: %[[TMP_2:.*]] = insertvalue { double, double } undef, double %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { double, double } %[[TMP_2]], double 0.000000e+00, 1
// LLVM-NEXT: store { double, double } %[[COMPLEX]], ptr {{.*}}, align 8

// OGCG: %[[TMP:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: %[[REAL:.*]] = sitofp i32 %[[TMP]] to double
// OGCG: store double %[[REAL]], ptr {{.*}}, align 8
// OGCG: store double 0.000000e+00, ptr getelementptr inbounds nuw ({ double, double }, ptr {{.*}}, i32 0, i32 1), align 8

// CIR: %[[FP_TO_INT:.*]] = cir.cast(float_to_int, %{{.*}} : !cir.double), !s32i
// CIR: %[[INT_TO_COMPLEX:.*]] = cir.cast(int_to_complex, %[[FP_TO_INT]] : !s32i), !cir.complex<!s32i>

//      LLVM: %[[TMP:.*]] = load double, ptr {{.*}}, align 8
// LLVM-NEXT: %[[REAL:.*]] = fptosi double %[[TMP]] to i32
// LLVM-NEXT: %[[TMP_2:.*]] = insertvalue { i32, i32 } undef, i32 %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { i32, i32 } %[[TMP_2]], i32 0, 1
// LLVM-NEXT:  store { i32, i32 } %[[COMPLEX]], ptr {{.*}}, align 4

// OGCG: %[[TMP:.*]] = load double, ptr {{.*}}, align 8
// OGCG: %[[REAL:.*]] = fptosi double %[[TMP]] to i32
// OGCG: store i32 %[[REAL]], ptr {{.*}}, align 4
// OGCG: store i32 0, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr {{.*}}, i32 0, i32 1), align 4

void scalar_to_complex_explicit() {
  cd = (double _Complex)sd;
  ci = (int _Complex)si;
  cd = (double _Complex)si;
  ci = (int _Complex)sd;
}

// CIR: %[[FP_TO_COMPLEX:.*]] = cir.cast(float_to_complex, %{{.*}} : !cir.double), !cir.complex<!cir.double>

//      LLVM: %[[REAL:.*]] = load double, ptr {{.*}}, align 8
// LLVM-NEXT: %[[TMP:.*]] = insertvalue { double, double } undef, double %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { double, double } %[[TMP]], double 0.000000e+00, 1
// LLVM-NEXT: store { double, double } %[[COMPLEX]], ptr {{.*}}, align 8

// OGCG: %[[REAL:.*]] = load double, ptr {{.*}}, align 8
// OGCG: store double %[[REAL]], ptr {{.*}}, align 8
// OGCG: store double 0.000000e+00, ptr getelementptr inbounds nuw ({ double, double }, ptr @cd, i32 0, i32 1), align 8

// CIR: %[[INT_TO_COMPLEX:.*]] = cir.cast(int_to_complex, %{{.*}} : !s32i), !cir.complex<!s32i>

//      LLVM: %[[REAL:.*]] = load i32, ptr {{.*}}, align 4
// LLVM-NEXT: %[[TMP:.*]] = insertvalue { i32, i32 } undef, i32 %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { i32, i32 } %[[TMP]], i32 0, 1
// LLVM-NEXT: store { i32, i32 } %[[COMPLEX]], ptr {{.*}}, align 4

// OGCG:  %[[REAL:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: store i32 %[[REAL]], ptr {{.*}}, align 4
// OGCG: store i32 0, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @ci, i32 0, i32 1), align 4

// CIR: %[[INT_TO_FP:.*]] = cir.cast(int_to_float, %{{.*}} : !s32i), !cir.double
// CIR: %[[FP_TO_COMPLEX:.*]] = cir.cast(float_to_complex, %[[INT_TO_FP]] : !cir.double), !cir.complex<!cir.double>

//      LLVM: %[[TMP:.*]] = load i32, ptr {{.*}}, align 4
// LLVM-NEXT: %[[REAL:.*]] = sitofp i32 %[[TMP]] to double
// LLVM-NEXT: %[[TMP_2:.*]] = insertvalue { double, double } undef, double %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { double, double } %[[TMP_2]], double 0.000000e+00, 1
// LLVM-NEXT: store { double, double } %[[COMPLEX]], ptr {{.*}}, align 8

// OGCG: %[[TMP:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: %[[REAL:.*]] = sitofp i32 %[[TMP]] to double
// OGCG: store double %[[REAL]], ptr {{.*}}, align 8
// OGCG: store double 0.000000e+00, ptr getelementptr inbounds nuw ({ double, double }, ptr {{.*}}, i32 0, i32 1), align 8

// CIR: %[[FP_TO_INT:.*]] = cir.cast(float_to_int, %{{.*}} : !cir.double), !s32i
// CIR: %[[INT_TO_COMPLEX:.*]] = cir.cast(int_to_complex, %[[FP_TO_INT]] : !s32i), !cir.complex<!s32i>

//      LLVM: %[[TMP:.*]] = load double, ptr {{.*}}, align 8
// LLVM-NEXT: %[[REAL:.*]] = fptosi double %[[TMP]] to i32
// LLVM-NEXT: %[[TMP_2:.*]] = insertvalue { i32, i32 } undef, i32 %[[REAL]], 0
// LLVM-NEXT: %[[COMPLEX:.*]] = insertvalue { i32, i32 } %[[TMP_2]], i32 0, 1
// LLVM-NEXT:  store { i32, i32 } %[[COMPLEX]], ptr {{.*}}, align 4

// OGCG: %[[TMP:.*]] = load double, ptr {{.*}}, align 8
// OGCG: %[[REAL:.*]] = fptosi double %[[TMP]] to i32
// OGCG: store i32 %[[REAL]], ptr {{.*}}, align 4
// OGCG: store i32 0, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr {{.*}}, i32 0, i32 1), align 4

void complex_to_scalar() {
  sd = (double)cd;
  si = (int)ci;
  sd = (double)ci;
  si = (int)cd;
}

// CIR: %[[FP_TO_COMPLEX_REAL:.*]] = cir.cast(float_complex_to_real, %{{.*}} : !cir.complex<!cir.double>), !cir.double

// LLVM: %[[REAL:.*]] = extractvalue { double, double } %{{.*}}, 0
// LLVM: store double %[[REAL]], ptr {{.*}}, align 8

// OGCG: %[[REAL:.*]] = load double, ptr {{.*}}, align 8
// OGCG: store double %[[REAL]], ptr {{.*}}, align 8

// CIR: %[[INT_COMPLEX_TO_REAL:.*]] = cir.cast(int_complex_to_real, %{{.*}} : !cir.complex<!s32i>), !s32i

// LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %{{.*}}, 0
// LLVM: store i32 %[[REAL]], ptr {{.*}}, align 4

// OGCG: %[[REAL:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: store i32 %[[REAL]], ptr {{.*}}, align 4

// CIR: %[[INT_COMPLEX_TO_REAL:.*]] = cir.cast(int_complex_to_real, %{{.*}} : !cir.complex<!s32i>), !s32i
// CIR: %[[INT_TO_FP:.*]] = cir.cast(int_to_float, %[[INT_COMPLEX_TO_REAL]] : !s32i), !cir.double

//      LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %[[REAL_TO_DOUBLE:.*]] = sitofp i32 %[[REAL]] to double
// LLVM-NEXT: store double %[[REAL_TO_DOUBLE]], ptr {{.*}}, align 8

// OGCG: %[[REAL:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: %[[INT_TO_FP:.*]] = sitofp i32 %[[REAL]] to double
// OGCG: store double %[[INT_TO_FP]], ptr {{.*}}, align 8

// CIR: %[[FP_TO_COMPLEX_REAL:.*]] = cir.cast(float_complex_to_real, %{{.*}} : !cir.complex<!cir.double>), !cir.double
// CIR: %[[FP_TO_INT:.*]] = cir.cast(float_to_int, %[[FP_TO_COMPLEX_REAL]] : !cir.double), !s32i

//      LLVM: %[[REAL:.*]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %[[REAL_TO_INT:.*]] = fptosi double %[[REAL]] to i32
// LLVM-NEXT: store i32 %[[REAL_TO_INT]], ptr {{.*}}, align 4

// OGCG: %[[REAL:.*]] = load double, ptr {{.*}}, align 8
// OGCG: %[[FP_TO_INT:.*]] = fptosi double %[[REAL]] to i32
// OGCG: store i32 %[[FP_TO_INT]], ptr {{.*}}, align 4

void complex_to_bool() {
  b = (bool)cd;
  b = (bool)ci;
}

// CIR: %[[FP_COMPLEX_TO_BOOL:.*]] = cir.cast(float_complex_to_bool, %{{.*}} : !cir.complex<!cir.double>), !cir.bool

//      LLVM: %[[REAL:.*]] = extractvalue { double, double } %{{.*}}, 0
// LLVM-NEXT: %[[IMAG:.*]] = extractvalue { double, double } %{{.*}}, 1
// LLVM-NEXT: %[[REAL_TO_BOOL:.*]] = fcmp une double %[[REAL]], 0.000000e+00
// LLVM-NEXT: %[[IMAG_TO_BOOL:.*]] = fcmp une double %[[IMAG]], 0.000000e+00
// LLVM-NEXT: %[[OR:.*]] = or i1 %[[REAL_TO_BOOL]], %[[IMAG_TO_BOOL]]
// LLVM-NEXT: %[[RESULT:.*]] = zext i1 %[[OR]] to i8
// LLVM-NEXT: store i8 %[[RESULT]], ptr {{.*}}, align 1

// OGCG: %[[REAL:.*]] = load double, ptr {{.*}}, align 8
// OGCG: %[[IMAG:.*]] = load double, ptr getelementptr inbounds nuw ({ double, double }, ptr {{.*}}, i32 0, i32 1), align 8
// OGCG: %[[REAL_TO_BOOL:.*]] = fcmp une double %[[REAL]], 0.000000e+00
// OGCG: %[[IMAG_TO_BOOL:.*]] = fcmp une double %[[IMAG]], 0.000000e+00
// OGCG: %[[COMPLEX_TO_BOOL:.*]] = or i1 %[[REAL_TO_BOOL]], %[[IMAG_TO_BOOL]]
// OGCG: %[[BOOL_TO_INT:.*]] = zext i1 %[[COMPLEX_TO_BOOL]] to i8
// OGCG: store i8 %[[BOOL_TO_INT]], ptr {{.*}}, align 1

// CIR: %[[INT_COMPLEX_TO_BOOL:.*]] = cir.cast(int_complex_to_bool, %{{.*}} : !cir.complex<!s32i>), !cir.bool

//      LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %{{.*}}, 0
// LLVM-NEXT: %[[IMAG:.*]] = extractvalue { i32, i32 } %{{.*}}, 1
// LLVM-NEXT: %[[REAL_TO_BOOL:.*]] = icmp ne i32 %[[REAL]], 0
// LLVM-NEXT: %[[IMAG_TO_BOOL:.*]] = icmp ne i32 %[[IMAG]], 0
// LLVM-NEXT: %[[OR:.*]] = or i1 %[[REAL_TO_BOOL]], %[[IMAG_TO_BOOL]]
// LLVM-NEXT: %[[RESULT:.*]] = zext i1 %[[OR]] to i8
// LLVM-NEXT: store i8 %[[RESULT]], ptr {{.*}}, align 1

// OGCG: %[[REAL:.*]] = load i32, ptr {{.*}}, align 4
// OGCG: %[[IMAG:.*]] = load i32, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr {{.*}}, i32 0, i32 1), align 4
// OGCG: %[[REAL_TO_BOOL:.*]] = icmp ne i32 %[[REAL]], 0
// OGCG: %[[IMAG_TO_BOOL:.*]] = icmp ne i32 %[[IMAG]], 0
// OGCG: %[[COMPLEX_TO_BOOL:.*]] = or i1 %[[REAL_TO_BOOL]], %[[IMAG_TO_BOOL]]
// OGCG: %[[BOOL_TO_INT:.*]] = zext i1 %[[COMPLEX_TO_BOOL]] to i8
// OGCG: store i8 %[[BOOL_TO_INT]], ptr {{.*}}, align 1

void complex_to_complex_cast() {
  cd = cf;
  ci = cs;
}

// CIR: %[[TMP:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[FP_COMPLEX:.*]] = cir.cast(float_complex, %[[TMP]] : !cir.complex<!cir.float>), !cir.complex<!cir.double>

// LLVM: %[[REAL:.*]] = extractvalue { float, float } %{{.*}}, 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %{{.*}}, 1
// LLVM: %[[REAL_FP_CAST:.*]] = fpext float %[[REAL]] to double
// LLVM: %[[IMAG_FP_CAST:.*]] = fpext float %[[IMAG]] to double
// LLVM: %[[TMP:.*]] = insertvalue { double, double } undef, double %[[REAL_FP_CAST]], 0
// LLVM: %[[COMPLEX:.*]] = insertvalue { double, double } %[[TMP]], double %[[IMAG_FP_CAST]], 1
// LLVM: store { double, double } %[[COMPLEX]], ptr {{.*}}, align 8

// OGCG: %[[REAL:.*]] = load float, ptr {{.*}}, align 4
// OGCG: %[[IMAG:.*]] = load float, ptr getelementptr inbounds nuw ({ float, float }, ptr {{.*}}, i32 0, i32 1), align 4
// OGCG: %[[REAL_FP_CAST:.*]] = fpext float %[[REAL]] to double
// OGCG: %[[IMAG_FP_CAST:.*]] = fpext float %[[IMAG]] to double
// OGCG: store double %[[REAL_FP_CAST]], ptr {{.*}}, align 8
// OGCG: store double %[[IMAG_FP_CAST]], ptr getelementptr inbounds nuw ({ double, double }, ptr {{.*}}, i32 0, i32 1), align 8

// CIR: %[[TMP:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.complex<!s16i>>, !cir.complex<!s16i>
// CIR: %[[INT_COMPLEX:.*]] = cir.cast(int_complex, %[[TMP]] : !cir.complex<!s16i>), !cir.complex<!s32i>

// LLVM: %[[REAL:.*]] = extractvalue { i16, i16 } %{{.*}}, 0
// LLVM: %[[IMAG:.*]] = extractvalue { i16, i16 } %{{.*}}, 1
// LLVM: %[[REAL_INT_CAST:.*]] = sext i16 %[[REAL]] to i32
// LLVM: %[[IMAG_INT_CAST:.*]] = sext i16 %[[IMAG]] to i32
// LLVM: %[[TMP:.*]] = insertvalue { i32, i32 } undef, i32 %[[REAL_INT_CAST]], 0
// LLVM: %[[COMPLEX:.*]] = insertvalue { i32, i32 } %[[TMP]], i32 %[[IMAG_INT_CAST]], 1
// LLVM: store { i32, i32 } %[[COMPLEX]], ptr {{.*}}, align 4

// OGCG:  %[[REAL:.*]] = load i16, ptr {{.*}}, align 2
// OGCG: %[[IMAG:.*]] = load i16, ptr getelementptr inbounds nuw ({ i16, i16 }, ptr {{.*}}, i32 0, i32 1), align 2
// OGCG: %[[REAL_INT_CAST:.*]] = sext i16 %[[REAL]] to i32
// OGCG: %[[IMAG_INT_CAST:.*]] = sext i16 %[[IMAG]] to i32
// OGCG: store i32 %[[REAL_INT_CAST]], ptr {{.*}}, align 4
// OGCG: store i32 %[[IMAG_INT_CAST]], ptr getelementptr inbounds nuw ({ i32, i32 }, ptr {{.*}}, i32 0, i32 1), align 4

struct CX {
  double real;
  double imag;
};

void lvalue_to_rvalue_bitcast() {
   CX a;
   double _Complex b = __builtin_bit_cast(double _Complex, a);
}

// CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !cir.ptr<!rec_CX>), !cir.ptr<!cir.complex<!cir.double>>

// LLVM: %[[PTR_ADDR:.*]] = alloca %struct.CX, i64 1, align 8
// LLVM: %[[COMPLEX_ADDR:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[PTR_TO_COMPLEX:.*]] = load { double, double }, ptr %[[PTR_ADDR]], align 8
// LLVM: store { double, double } %[[PTR_TO_COMPLEX]], ptr %[[COMPLEX_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca %struct.CX, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { double, double }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load double, ptr %[[A_REAL_PTR]], align 8
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load double, ptr %[[A_IMAG_PTR]], align 8
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store double %[[A_REAL]], ptr %[[B_REAL_PTR]], align 8
// OGCG: store double %[[A_IMAG]], ptr %[[B_IMAG_PTR]], align 8

void lvalue_bitcast() {
  CX a;
  (double _Complex &)a = {};
}

// CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !cir.ptr<!rec_CX>), !cir.ptr<!cir.complex<!cir.double>>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.CX, i64 1, align 8
// LLVM: store { double, double } zeroinitializer, ptr %[[A_ADDR]], align 8

// OGCG: %[[A_ADDR]] = alloca %struct.CX, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store double 0.000000e+00, ptr %[[A_REAL_PTR]], align 8
// OGCG: store double 0.000000e+00, ptr %[[A_IMAG_PTR]], align 8

