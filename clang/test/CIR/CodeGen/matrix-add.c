// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-matrix -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-matrix -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-matrix -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
typedef float fx2x3_t __attribute__((matrix_type(2, 3)));
typedef int ix9x3_t __attribute__((matrix_type(9, 3)));
typedef unsigned long long ullx4x2_t __attribute__((matrix_type(4, 2)));

void add_matrix_matrix_double() {
  dx5x5_t a;
  dx5x5_t b;
  dx5x5_t c;
  a = b + c;
}

// CIR-LABEL: cir.func {{.*}}@add_matrix_matrix_double
// CIR: %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[C:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[RES:.*]] = cir.fadd %[[B]], %[[C]] : !cir.matrix<5 x 5 x !cir.double>
// CIR: cir.store {{.*}} %[[RES]], {{.*}} : !cir.matrix<5 x 5 x !cir.double>, !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>

// LLVM-LABEL: define {{.*}}void @add_matrix_matrix_double(
// LLVM: %[[B:.*]] = load <25 x double>, ptr {{.*}}, align 8
// LLVM: %[[C:.*]] = load <25 x double>, ptr {{.*}}, align 8
// LLVM: %[[RES:.*]] = fadd <25 x double> %[[B]], %[[C]]
// LLVM: store <25 x double> %[[RES]], ptr {{.*}}, align 8

void add_compound_assign_matrix_double() {
  dx5x5_t a;
  dx5x5_t b;
  a += b;
}

// CIR-LABEL: cir.func {{.*}}@add_compound_assign_matrix_double
// CIR: %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[RES:.*]] = cir.fadd %[[A]], %[[B]] : !cir.matrix<5 x 5 x !cir.double>
// CIR: cir.store {{.*}} %[[RES]], {{.*}} : !cir.matrix<5 x 5 x !cir.double>, !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>

// LLVM-LABEL: define {{.*}}void @add_compound_assign_matrix_double(
// LLVM: %[[B:.*]] = load <25 x double>, ptr {{.*}}, align 8
// LLVM: %[[A:.*]] = load <25 x double>, ptr {{.*}}, align 8
// LLVM: %[[RES:.*]] = fadd <25 x double> %[[A]], %[[B]]
// LLVM: store <25 x double> %[[RES]], ptr {{.*}}, align 8

void add_matrix_matrix_float() {
  fx2x3_t a;
  fx2x3_t b;
  fx2x3_t c;
  a = b + c;
}

// CIR-LABEL: cir.func {{.*}}@add_matrix_matrix_float
// CIR: %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<2 x 3 x !cir.float>>, !cir.matrix<2 x 3 x !cir.float>
// CIR: %[[C:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<2 x 3 x !cir.float>>, !cir.matrix<2 x 3 x !cir.float>
// CIR: %[[RES:.*]] = cir.fadd %[[B]], %[[C]] : !cir.matrix<2 x 3 x !cir.float>

// LLVM-LABEL: define {{.*}}void @add_matrix_matrix_float(
// LLVM: %[[B:.*]] = load <6 x float>, ptr {{.*}}, align 4
// LLVM: %[[C:.*]] = load <6 x float>, ptr {{.*}}, align 4
// LLVM: %[[RES:.*]] = fadd <6 x float> %[[B]], %[[C]]
// LLVM: store <6 x float> %[[RES]], ptr {{.*}}, align 4

void add_matrix_matrix_int() {
  ix9x3_t a;
  ix9x3_t b;
  ix9x3_t c;
  a = b + c;
}

// CIR-LABEL: cir.func {{.*}}@add_matrix_matrix_int
// CIR: %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<9 x 3 x !s32i>>, !cir.matrix<9 x 3 x !s32i>
// CIR: %[[C:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<9 x 3 x !s32i>>, !cir.matrix<9 x 3 x !s32i>
// CIR: %[[RES:.*]] = cir.add %[[B]], %[[C]] : !cir.matrix<9 x 3 x !s32i>
// CIR: cir.store {{.*}} %[[RES]], {{.*}} : !cir.matrix<9 x 3 x !s32i>, !cir.ptr<!cir.matrix<9 x 3 x !s32i>>

// LLVM-LABEL: define {{.*}}void @add_matrix_matrix_int(
// LLVM: %[[B:.*]] = load <27 x i32>, ptr {{.*}}, align 4
// LLVM: %[[C:.*]] = load <27 x i32>, ptr {{.*}}, align 4
// LLVM: %[[RES:.*]] = add <27 x i32> %[[B]], %[[C]]
// LLVM: store <27 x i32> %[[RES]], ptr {{.*}}, align 4

void add_compound_assign_matrix_int() {
  ix9x3_t a;
  ix9x3_t b;
  a += b;
}

// CIR-LABEL: cir.func {{.*}}@add_compound_assign_matrix_int
// CIR: %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<9 x 3 x !s32i>>, !cir.matrix<9 x 3 x !s32i>
// CIR: %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<9 x 3 x !s32i>>, !cir.matrix<9 x 3 x !s32i>
// CIR: %[[RES:.*]] = cir.add %[[A]], %[[B]] : !cir.matrix<9 x 3 x !s32i>

// LLVM-LABEL: define {{.*}}void @add_compound_assign_matrix_int(
// LLVM: %[[B:.*]] = load <27 x i32>, ptr {{.*}}, align 4
// LLVM: %[[A:.*]] = load <27 x i32>, ptr {{.*}}, align 4
// LLVM: %[[RES:.*]] = add <27 x i32> %[[A]], %[[B]]
// LLVM: store <27 x i32> %[[RES]], ptr {{.*}}, align 4

void add_matrix_matrix_unsigned() {
  ullx4x2_t a;
  ullx4x2_t b;
  ullx4x2_t c;
  a = b + c;
}

// CIR-LABEL: cir.func {{.*}}@add_matrix_matrix_unsigned
// CIR: %[[RES:.*]] = cir.add %{{.*}}, %{{.*}} : !cir.matrix<4 x 2 x !u64i>

// LLVM-LABEL: define {{.*}}void @add_matrix_matrix_unsigned(
// LLVM: %[[RES:.*]] = add <8 x i64> %{{.*}}, %{{.*}}

void add_matrix_scalar_double_double() {
  dx5x5_t a;
  double vd;
  a = a + vd;
}

// CIR-LABEL: cir.func {{.*}}@add_matrix_scalar_double_double
// CIR: %[[MAT:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[SCALAR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[SCALAR]] : !cir.double, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[RES:.*]] = cir.fadd %[[MAT]], %[[SPLAT]] : !cir.matrix<5 x 5 x !cir.double>

// LLVM-LABEL: define {{.*}}void @add_matrix_scalar_double_double(
// LLVM: %[[MAT:.*]] = load <25 x double>, ptr {{.*}}, align 8
// LLVM: %[[SCALAR:.*]] = load double, ptr {{.*}}, align 8
// LLVM: %[[EMBED:.*]] = insertelement <25 x double> poison, double %[[SCALAR]], i64 0
// LLVM: %[[SPLAT:.*]] = shufflevector <25 x double> %[[EMBED]], <25 x double> poison, <25 x i32> zeroinitializer
// LLVM: %[[RES:.*]] = fadd <25 x double> %[[MAT]], %[[SPLAT]]
// LLVM: store <25 x double> %[[RES]], ptr {{.*}}, align 8

void add_scalar_matrix_double_double() {
  dx5x5_t a;
  double vd;
  a = vd + a;
}

// CIR-LABEL: cir.func {{.*}}@add_scalar_matrix_double_double
// CIR: %[[SCALAR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: %[[MAT:.*]] = cir.load {{.*}} : !cir.ptr<!cir.matrix<5 x 5 x !cir.double>>, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[SCALAR]] : !cir.double, !cir.matrix<5 x 5 x !cir.double>
// CIR: %[[RES:.*]] = cir.fadd %[[SPLAT]], %[[MAT]] : !cir.matrix<5 x 5 x !cir.double>

// LLVM-LABEL: define {{.*}}void @add_scalar_matrix_double_double(
// LLVM: %[[SCALAR:.*]] = load double, ptr {{.*}}, align 8
// LLVM: %[[MAT:.*]] = load <25 x double>, ptr {{.*}}, align 8
// LLVM: %[[EMBED:.*]] = insertelement <25 x double> poison, double %[[SCALAR]], i64 0
// LLVM: %[[SPLAT:.*]] = shufflevector <25 x double> %[[EMBED]], <25 x double> poison, <25 x i32> zeroinitializer
// LLVM: %[[RES:.*]] = fadd <25 x double> %[[SPLAT]], %[[MAT]]

void add_matrix_scalar_double_float() {
  dx5x5_t a;
  float vf;
  a = a + vf;
}

// CIR-LABEL: cir.func {{.*}}@add_matrix_scalar_double_float
// CIR: %[[SCALAR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[EXT:.*]] = cir.cast floating %[[SCALAR]] : !cir.float -> !cir.double
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[EXT]] : !cir.double, !cir.matrix<5 x 5 x !cir.double>
// CIR: cir.fadd %{{.*}}, %[[SPLAT]] : !cir.matrix<5 x 5 x !cir.double>

// LLVM-LABEL: define {{.*}}void @add_matrix_scalar_double_float(
// LLVM: %[[SCALAR:.*]] = load float, ptr {{.*}}, align 4
// LLVM: %[[EXT:.*]] = fpext float %[[SCALAR]] to double
// LLVM: %[[EMBED:.*]] = insertelement <25 x double> poison, double %[[EXT]], i64 0
// LLVM: %[[SPLAT:.*]] = shufflevector <25 x double> %[[EMBED]], <25 x double> poison, <25 x i32> zeroinitializer
// LLVM: fadd <25 x double> %{{.*}}, %[[SPLAT]]

void add_matrix_scalar_int_long() {
  ix9x3_t a;
  long vl;
  a = a + vl;
}

// CIR-LABEL: cir.func {{.*}}@add_matrix_scalar_int_long
// CIR: %[[SCALAR:.*]] = cir.load {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: %[[TRUNC:.*]] = cir.cast integral %[[SCALAR]] : !s64i -> !s32i
// CIR: %[[SPLAT:.*]] = cir.vec.splat %[[TRUNC]] : !s32i, !cir.matrix<9 x 3 x !s32i>
// CIR: cir.add %{{.*}}, %[[SPLAT]] : !cir.matrix<9 x 3 x !s32i>

// LLVM-LABEL: define {{.*}}void @add_matrix_scalar_int_long(
// LLVM: %[[SCALAR:.*]] = load i64, ptr {{.*}}, align 8
// LLVM: %[[TRUNC:.*]] = trunc i64 %[[SCALAR]] to i32
// LLVM: %[[EMBED:.*]] = insertelement <27 x i32> poison, i32 %[[TRUNC]], i64 0
// LLVM: %[[SPLAT:.*]] = shufflevector <27 x i32> %[[EMBED]], <27 x i32> poison, <27 x i32> zeroinitializer
// LLVM: add <27 x i32> %{{.*}}, %[[SPLAT]]

void add_compound_matrix_scalar_float_float() {
  fx2x3_t b;
  float vf;
  b += vf;
}

// CIR-LABEL: cir.func {{.*}}@add_compound_matrix_scalar_float_float
// CIR: %[[SPLAT:.*]] = cir.vec.splat %{{.*}} : !cir.float, !cir.matrix<2 x 3 x !cir.float>
// CIR: cir.fadd %{{.*}}, %[[SPLAT]] : !cir.matrix<2 x 3 x !cir.float>

// LLVM-LABEL: define {{.*}}void @add_compound_matrix_scalar_float_float(
// LLVM: %[[EMBED:.*]] = insertelement <6 x float> poison, float %{{.*}}, i64 0
// LLVM: %[[SPLAT:.*]] = shufflevector <6 x float> %[[EMBED]], <6 x float> poison, <6 x i32> zeroinitializer
// LLVM: fadd <6 x float> %{{.*}}, %[[SPLAT]]
