// RUN: %clang_cc1 -no-enable-noundef-analysis -fenable-matrix -fclang-abi-compat=latest -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - -std=c++17 | FileCheck %s

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
typedef float fx3x4_t __attribute__((matrix_type(3, 4)));

// CHECK: %struct.Matrix = type { i8, [12 x float], float }

void load_store(dx5x5_t *a, dx5x5_t *b) {
  // CHECK-LABEL:  define{{.*}} void @_Z10load_storePu11matrix_typeILm5ELm5EdES0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %1 = load <25 x double>, ptr %0, align 8
  // CHECK-NEXT:    %2 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    store <25 x double> %1, ptr %2, align 8
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef float fx3x3_t __attribute__((matrix_type(3, 3)));

void parameter_passing(fx3x3_t a, fx3x3_t *b) {
  // CHECK-LABEL: define{{.*}} void @_Z17parameter_passingu11matrix_typeILm3ELm3EfEPS_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [9 x float], align 4
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store <9 x float> %a, ptr %a.addr, align 4
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load <9 x float>, ptr %a.addr, align 4
  // CHECK-NEXT:    %1 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    store <9 x float> %0, ptr %1, align 4
  // CHECK-NEXT:    ret void
  *b = a;
}

fx3x3_t return_matrix(fx3x3_t *a) {
  // CHECK-LABEL: define{{.*}} <9 x float> @_Z13return_matrixPu11matrix_typeILm3ELm3EfE(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    %1 = load <9 x float>, ptr %0, align 4
  // CHECK-NEXT:    ret <9 x float> %1
  return *a;
}

struct Matrix {
  char Tmp1;
  fx3x4_t Data;
  float Tmp2;
};

void matrix_struct_pointers(Matrix *a, Matrix *b) {
  // CHECK-LABEL: define{{.*}} void @_Z22matrix_struct_pointersP6MatrixS0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %struct.Matrix, ptr %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = load <12 x float>, ptr %Data, align 4
  // CHECK-NEXT:    %2 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %struct.Matrix, ptr %2, i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> %1, ptr %Data1, align 4
  // CHECK-NEXT:    ret void
  b->Data = a->Data;
}

void matrix_struct_reference(Matrix &a, Matrix &b) {
  // CHECK-LABEL: define{{.*}} void @_Z23matrix_struct_referenceR6MatrixS0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %struct.Matrix, ptr %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = load <12 x float>, ptr %Data, align 4
  // CHECK-NEXT:    %2 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %struct.Matrix, ptr %2, i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> %1, ptr %Data1, align 4
  // CHECK-NEXT:    ret void
  b.Data = a.Data;
}

class MatrixClass {
public:
  int Tmp1;
  fx3x4_t Data;
  long Tmp2;
};

void matrix_class_reference(MatrixClass &a, MatrixClass &b) {
  // CHECK-LABEL: define{{.*}} void @_Z22matrix_class_referenceR11MatrixClassS0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %class.MatrixClass, ptr %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = load <12 x float>, ptr %Data, align 4
  // CHECK-NEXT:    %2 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClass, ptr %2, i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> %1, ptr %Data1, align 4
  // CHECK-NEXT:    ret void
  b.Data = a.Data;
}

template <typename Ty, unsigned Rows, unsigned Cols>
class MatrixClassTemplate {
public:
  using MatrixTy = Ty __attribute__((matrix_type(Rows, Cols)));
  int Tmp1;
  MatrixTy Data;
  long Tmp2;
};

template <typename Ty, unsigned Rows, unsigned Cols>
void matrix_template_reference(MatrixClassTemplate<Ty, Rows, Cols> &a, MatrixClassTemplate<Ty, Rows, Cols> &b) {
  b.Data = a.Data;
}

MatrixClassTemplate<float, 10, 15> matrix_template_reference_caller(float *Data) {
  // CHECK-LABEL: define{{.*}} void @_Z32matrix_template_reference_callerPf(ptr dead_on_unwind noalias writable sret(%class.MatrixClassTemplate) align 8 %agg.result, ptr %Data
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %Data.addr = alloca ptr, align 8
  // CHECK-NEXT:    %Arg = alloca %class.MatrixClassTemplate, align 8
  // CHECK-NEXT:    store ptr %Data, ptr %Data.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %Data.addr, align 8
  // CHECK-NEXT:    %1 = load <150 x float>, ptr %0, align 4
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr %Arg, i32 0, i32 1
  // CHECK-NEXT:    store <150 x float> %1, ptr %Data1, align 4
  // CHECK-NEXT:    call void @_Z25matrix_template_referenceIfLj10ELj15EEvR19MatrixClassTemplateIT_XT0_EXT1_EES3_(ptr nonnull align 8 dereferenceable(616) %Arg, ptr nonnull align 8 dereferenceable(616) %agg.result)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr void @_Z25matrix_template_referenceIfLj10ELj15EEvR19MatrixClassTemplateIT_XT0_EXT1_EES3_(ptr nonnull align 8 dereferenceable(616) %a, ptr nonnull align 8 dereferenceable(616) %b)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = load <150 x float>, ptr %Data, align 4
  // CHECK-NEXT:    %2 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr %2, i32 0, i32 1
  // CHECK-NEXT:    store <150 x float> %1, ptr %Data1, align 4
  // CHECK-NEXT:    ret void

  MatrixClassTemplate<float, 10, 15> Result, Arg;
  Arg.Data = *((MatrixClassTemplate<float, 10, 15>::MatrixTy *)Data);
  matrix_template_reference(Arg, Result);
  return Result;
}

template <class T, unsigned long R, unsigned long C>
using matrix = T __attribute__((matrix_type(R, C)));

template <int N>
struct selector {};

template <class T, unsigned long R, unsigned long C>
selector<0> use_matrix(matrix<T, R, C> &m) {}

template <class T, unsigned long R>
selector<1> use_matrix(matrix<T, R, 10> &m) {}

template <class T>
selector<2> use_matrix(matrix<T, 10, 10> &m) {}

template <class T, unsigned long C>
selector<3> use_matrix(matrix<T, 10, C> &m) {}

template <unsigned long R, unsigned long C>
selector<4> use_matrix(matrix<float, R, C> &m) {}

void test_template_deduction() {

  // CHECK-LABEL: define{{.*}} void @_Z23test_template_deductionv()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m0 = alloca [120 x i32], align 4
  // CHECK-NEXT:    %w = alloca %struct.selector, align 1
  // CHECK-NEXT:    %undef.agg.tmp = alloca %struct.selector, align 1
  // CHECK-NEXT:    %m1 = alloca [100 x i32], align 4
  // CHECK-NEXT:    %x = alloca %struct.selector.0, align 1
  // CHECK-NEXT:    %undef.agg.tmp1 = alloca %struct.selector.0, align 1
  // CHECK-NEXT:    %m2 = alloca [120 x i32], align 4
  // CHECK-NEXT:    %y = alloca %struct.selector.1, align 1
  // CHECK-NEXT:    %undef.agg.tmp2 = alloca %struct.selector.1, align 1
  // CHECK-NEXT:    %m3 = alloca [144 x i32], align 4
  // CHECK-NEXT:    %z = alloca %struct.selector.2, align 1
  // CHECK-NEXT:    %undef.agg.tmp3 = alloca %struct.selector.2, align 1
  // CHECK-NEXT:    %m4 = alloca [144 x float], align 4
  // CHECK-NEXT:    %v = alloca %struct.selector.3, align 1
  // CHECK-NEXT:    %undef.agg.tmp4 = alloca %struct.selector.3, align 1
  // CHECK-NEXT:    call void @_Z10use_matrixIiLm12EE8selectorILi3EERu11matrix_typeILm10EXT0_ET_E(ptr nonnull align 4 dereferenceable(480) %m0)
  // CHECK-NEXT:    call void @_Z10use_matrixIiE8selectorILi2EERu11matrix_typeILm10ELm10ET_E(ptr nonnull align 4 dereferenceable(400) %m1)
  // CHECK-NEXT:    call void @_Z10use_matrixIiLm12EE8selectorILi1EERu11matrix_typeIXT0_ELm10ET_E(ptr nonnull align 4 dereferenceable(480) %m2)
  // CHECK-NEXT:    call void @_Z10use_matrixIiLm12ELm12EE8selectorILi0EERu11matrix_typeIXT0_EXT1_ET_E(ptr nonnull align 4 dereferenceable(576) %m3)
  // CHECK-NEXT:    call void @_Z10use_matrixILm12ELm12EE8selectorILi4EERu11matrix_typeIXT_EXT0_EfE(ptr nonnull align 4 dereferenceable(576) %m4)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiLm12EE8selectorILi3EERu11matrix_typeILm10EXT0_ET_E(ptr nonnull align 4 dereferenceable(480) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiE8selectorILi2EERu11matrix_typeILm10ELm10ET_E(ptr nonnull align 4 dereferenceable(400) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiLm12EE8selectorILi1EERu11matrix_typeIXT0_ELm10ET_E(ptr nonnull align 4 dereferenceable(480) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiLm12ELm12EE8selectorILi0EERu11matrix_typeIXT0_EXT1_ET_E(ptr nonnull align 4 dereferenceable(576) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixILm12ELm12EE8selectorILi4EERu11matrix_typeIXT_EXT0_EfE(ptr nonnull align 4 dereferenceable(576)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  matrix<int, 10, 12> m0;
  selector<3> w = use_matrix(m0);
  matrix<int, 10, 10> m1;
  selector<2> x = use_matrix(m1);
  matrix<int, 12, 10> m2;
  selector<1> y = use_matrix(m2);
  matrix<int, 12, 12> m3;
  selector<0> z = use_matrix(m3);
  matrix<float, 12, 12> m4;
  selector<4> v = use_matrix(m4);
}

template <auto R>
void foo(matrix<int, R, 10> &m) {
}

void test_auto_t() {
  // CHECK-LABEL: define{{.*}} void @_Z11test_auto_tv()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m = alloca [130 x i32], align 4
  // CHECK-NEXT:    call void @_Z3fooITnDaLm13EEvRu11matrix_typeIXT_ELm10EiE(ptr nonnull align 4 dereferenceable(520) %m)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr void @_Z3fooITnDaLm13EEvRu11matrix_typeIXT_ELm10EiE(ptr nonnull align 4 dereferenceable(520) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    ret void

  matrix<int, 13, 10> m;
  foo(m);
}

template <unsigned long R, unsigned long C>
matrix<float, R + 1, C + 2> use_matrix_2(matrix<int, R, C> &m) {}

template <unsigned long R, unsigned long C>
selector<0> use_matrix_2(matrix<int, R + 2, C / 2> &m1, matrix<float, R, C> &m2) {}

template <unsigned long R, unsigned long C>
selector<1> use_matrix_2(matrix<int, R + C, C> &m1, matrix<float, R, C - R> &m2) {}

template <unsigned long R>
matrix<float, R + R, R - 3> use_matrix_2(matrix<int, R, 10> &m1) {}

template <unsigned long R>
selector<2> use_matrix_3(matrix<int, R - 2, R> &m) {}

void test_use_matrix_2() {
  // CHECK-LABEL: define{{.*}} void @_Z17test_use_matrix_2v()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m1 = alloca [24 x i32], align 4
  // CHECK-NEXT:    %r1 = alloca [40 x float], align 4
  // CHECK-NEXT:    %m2 = alloca [24 x float], align 4
  // CHECK-NEXT:    %r2 = alloca %struct.selector.2, align 1
  // CHECK-NEXT:    %undef.agg.tmp = alloca %struct.selector.2, align 1
  // CHECK-NEXT:    %m3 = alloca [104 x i32], align 4
  // CHECK-NEXT:    %m4 = alloca [15 x float], align 4
  // CHECK-NEXT:    %r3 = alloca %struct.selector.1, align 1
  // CHECK-NEXT:    %undef.agg.tmp1 = alloca %struct.selector.1, align 1
  // CHECK-NEXT:    %m5 = alloca [50 x i32], align 4
  // CHECK-NEXT:    %r4 = alloca [20 x float], align 4
  // CHECK-NEXT:    %r5 = alloca %struct.selector.0, align 1
  // CHECK-NEXT:    %undef.agg.tmp3 = alloca %struct.selector.0, align 1
  // CHECK-NEXT:    %call = call <40 x float> @_Z12use_matrix_2ILm4ELm6EEu11matrix_typeIXplT_Li1EEXplT0_Li2EEfERu11matrix_typeIXT_EXT0_EiE(ptr nonnull align 4 dereferenceable(96) %m1)
  // CHECK-NEXT:    store <40 x float> %call, ptr %r1, align 4
  // CHECK-NEXT:    call void @_Z12use_matrix_2ILm2ELm12EE8selectorILi0EERu11matrix_typeIXplT_Li2EEXdvT0_Li2EEiERu11matrix_typeIXT_EXT0_EfE(ptr nonnull align 4 dereferenceable(96) %m1, ptr nonnull align 4 dereferenceable(96) %m2)
  // CHECK-NEXT:    call void @_Z12use_matrix_2ILm5ELm8EE8selectorILi1EERu11matrix_typeIXplT_T0_EXT0_EiERu11matrix_typeIXT_EXmiT0_T_EfE(ptr nonnull align 4 dereferenceable(416) %m3, ptr nonnull align 4 dereferenceable(60) %m4)
  // CHECK-NEXT:    %call2 = call <20 x float> @_Z12use_matrix_2ILm5EEu11matrix_typeIXplT_T_EXmiT_Li3EEfERu11matrix_typeIXT_ELm10EiE(ptr nonnull align 4 dereferenceable(200) %m5)
  // CHECK-NEXT:    store <20 x float> %call2, ptr %r4, align 4
  // CHECK-NEXT:    call void @_Z12use_matrix_3ILm6EE8selectorILi2EERu11matrix_typeIXmiT_Li2EEXT_EiE(ptr nonnull align 4 dereferenceable(96) %m1)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr <40 x float> @_Z12use_matrix_2ILm4ELm6EEu11matrix_typeIXplT_Li1EEXplT0_Li2EEfERu11matrix_typeIXT_EXT0_EiE(ptr nonnull align 4 dereferenceable(96) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z12use_matrix_2ILm2ELm12EE8selectorILi0EERu11matrix_typeIXplT_Li2EEXdvT0_Li2EEiERu11matrix_typeIXT_EXT0_EfE(ptr nonnull align 4 dereferenceable(96) %m1, ptr nonnull align 4 dereferenceable(96) %m2)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m1.addr = alloca ptr, align 8
  // CHECK-NEXT:    %m2.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m1, ptr %m1.addr, align 8
  // CHECK-NEXT:    store ptr %m2, ptr %m2.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z12use_matrix_2ILm5ELm8EE8selectorILi1EERu11matrix_typeIXplT_T0_EXT0_EiERu11matrix_typeIXT_EXmiT0_T_EfE(ptr nonnull align 4 dereferenceable(416) %m1, ptr nonnull align 4 dereferenceable(60) %m2)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m1.addr = alloca ptr, align 8
  // CHECK-NEXT:    %m2.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m1, ptr %m1.addr, align 8
  // CHECK-NEXT:    store ptr %m2, ptr %m2.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr <20 x float> @_Z12use_matrix_2ILm5EEu11matrix_typeIXplT_T_EXmiT_Li3EEfERu11matrix_typeIXT_ELm10EiE(ptr nonnull align 4 dereferenceable(200) %m1)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m1.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m1, ptr %m1.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z12use_matrix_3ILm6EE8selectorILi2EERu11matrix_typeIXmiT_Li2EEXT_EiE(ptr nonnull align 4 dereferenceable(96) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %m, ptr %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  matrix<int, 4, 6> m1;
  matrix<float, 5, 8> r1 = use_matrix_2(m1);

  matrix<float, 2, 12> m2;
  selector<0> r2 = use_matrix_2(m1, m2);

  matrix<int, 13, 8> m3;
  matrix<float, 5, 3> m4;
  selector<1> r3 = use_matrix_2(m3, m4);

  matrix<int, 5, 10> m5;
  matrix<float, 10, 2> r4 = use_matrix_2(m5);

  selector<2> r5 = use_matrix_3(m1);
}
