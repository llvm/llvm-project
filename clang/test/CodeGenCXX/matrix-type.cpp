// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - -std=c++17 | FileCheck %s

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));
typedef float fx3x4_t __attribute__((matrix_type(3, 4)));

// CHECK: %struct.Matrix = type { i8, [12 x float], float }

void load_store(dx5x5_t *a, dx5x5_t *b) {
  // CHECK-LABEL:  define void @_Z10load_storePU11matrix_typeLm5ELm5EdS0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [25 x double]*, align 8
  // CHECK-NEXT:    %b.addr = alloca [25 x double]*, align 8
  // CHECK-NEXT:    store [25 x double]* %a, [25 x double]** %a.addr, align 8
  // CHECK-NEXT:    store [25 x double]* %b, [25 x double]** %b.addr, align 8
  // CHECK-NEXT:    %0 = load [25 x double]*, [25 x double]** %b.addr, align 8
  // CHECK-NEXT:    %1 = bitcast [25 x double]* %0 to <25 x double>*
  // CHECK-NEXT:    %2 = load <25 x double>, <25 x double>* %1, align 8
  // CHECK-NEXT:    %3 = load [25 x double]*, [25 x double]** %a.addr, align 8
  // CHECK-NEXT:    %4 = bitcast [25 x double]* %3 to <25 x double>*
  // CHECK-NEXT:    store <25 x double> %2, <25 x double>* %4, align 8
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef float fx3x3_t __attribute__((matrix_type(3, 3)));

void parameter_passing(fx3x3_t a, fx3x3_t *b) {
  // CHECK-LABEL: define void @_Z17parameter_passingU11matrix_typeLm3ELm3EfPS_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [9 x float], align 4
  // CHECK-NEXT:    %b.addr = alloca [9 x float]*, align 8
  // CHECK-NEXT:    %0 = bitcast [9 x float]* %a.addr to <9 x float>*
  // CHECK-NEXT:    store <9 x float> %a, <9 x float>* %0, align 4
  // CHECK-NEXT:    store [9 x float]* %b, [9 x float]** %b.addr, align 8
  // CHECK-NEXT:    %1 = load <9 x float>, <9 x float>* %0, align 4
  // CHECK-NEXT:    %2 = load [9 x float]*, [9 x float]** %b.addr, align 8
  // CHECK-NEXT:    %3 = bitcast [9 x float]* %2 to <9 x float>*
  // CHECK-NEXT:    store <9 x float> %1, <9 x float>* %3, align 4
  // CHECK-NEXT:    ret void
  *b = a;
}

fx3x3_t return_matrix(fx3x3_t *a) {
  // CHECK-LABEL: define <9 x float> @_Z13return_matrixPU11matrix_typeLm3ELm3Ef(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [9 x float]*, align 8
  // CHECK-NEXT:    store [9 x float]* %a, [9 x float]** %a.addr, align 8
  // CHECK-NEXT:    %0 = load [9 x float]*, [9 x float]** %a.addr, align 8
  // CHECK-NEXT:    %1 = bitcast [9 x float]* %0 to <9 x float>*
  // CHECK-NEXT:    %2 = load <9 x float>, <9 x float>* %1, align 4
  // CHECK-NEXT:    ret <9 x float> %2
  return *a;
}

struct Matrix {
  char Tmp1;
  fx3x4_t Data;
  float Tmp2;
};

void matrix_struct_pointers(Matrix *a, Matrix *b) {
  // CHECK-LABEL: define void @_Z22matrix_struct_pointersP6MatrixS0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca %struct.Matrix*, align 8
  // CHECK-NEXT:    %b.addr = alloca %struct.Matrix*, align 8
  // CHECK-NEXT:    store %struct.Matrix* %a, %struct.Matrix** %a.addr, align 8
  // CHECK-NEXT:    store %struct.Matrix* %b, %struct.Matrix** %b.addr, align 8
  // CHECK-NEXT:    %0 = load %struct.Matrix*, %struct.Matrix** %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds %struct.Matrix, %struct.Matrix* %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = bitcast [12 x float]* %Data to <12 x float>*
  // CHECK-NEXT:    %2 = load <12 x float>, <12 x float>* %1, align 4
  // CHECK-NEXT:    %3 = load %struct.Matrix*, %struct.Matrix** %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds %struct.Matrix, %struct.Matrix* %3, i32 0, i32 1
  // CHECK-NEXT:    %4 = bitcast [12 x float]* %Data1 to <12 x float>*
  // CHECK-NEXT:    store <12 x float> %2, <12 x float>* %4, align 4
  // CHECK-NEXT:    ret void
  b->Data = a->Data;
}

void matrix_struct_reference(Matrix &a, Matrix &b) {
  // CHECK-LABEL: define void @_Z23matrix_struct_referenceR6MatrixS0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca %struct.Matrix*, align 8
  // CHECK-NEXT:    %b.addr = alloca %struct.Matrix*, align 8
  // CHECK-NEXT:    store %struct.Matrix* %a, %struct.Matrix** %a.addr, align 8
  // CHECK-NEXT:    store %struct.Matrix* %b, %struct.Matrix** %b.addr, align 8
  // CHECK-NEXT:    %0 = load %struct.Matrix*, %struct.Matrix** %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds %struct.Matrix, %struct.Matrix* %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = bitcast [12 x float]* %Data to <12 x float>*
  // CHECK-NEXT:    %2 = load <12 x float>, <12 x float>* %1, align 4
  // CHECK-NEXT:    %3 = load %struct.Matrix*, %struct.Matrix** %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds %struct.Matrix, %struct.Matrix* %3, i32 0, i32 1
  // CHECK-NEXT:    %4 = bitcast [12 x float]* %Data1 to <12 x float>*
  // CHECK-NEXT:    store <12 x float> %2, <12 x float>* %4, align 4
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
  // CHECK-LABEL: define void @_Z22matrix_class_referenceR11MatrixClassS0_(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca %class.MatrixClass*, align 8
  // CHECK-NEXT:    %b.addr = alloca %class.MatrixClass*, align 8
  // CHECK-NEXT:    store %class.MatrixClass* %a, %class.MatrixClass** %a.addr, align 8
  // CHECK-NEXT:    store %class.MatrixClass* %b, %class.MatrixClass** %b.addr, align 8
  // CHECK-NEXT:    %0 = load %class.MatrixClass*, %class.MatrixClass** %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds %class.MatrixClass, %class.MatrixClass* %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = bitcast [12 x float]* %Data to <12 x float>*
  // CHECK-NEXT:    %2 = load <12 x float>, <12 x float>* %1, align 4
  // CHECK-NEXT:    %3 = load %class.MatrixClass*, %class.MatrixClass** %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds %class.MatrixClass, %class.MatrixClass* %3, i32 0, i32 1
  // CHECK-NEXT:    %4 = bitcast [12 x float]* %Data1 to <12 x float>*
  // CHECK-NEXT:    store <12 x float> %2, <12 x float>* %4, align 4
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
  // CHECK-LABEL: define void @_Z32matrix_template_reference_callerPf(%class.MatrixClassTemplate* noalias sret align 8 %agg.result, float* %Data
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %Data.addr = alloca float*, align 8
  // CHECK-NEXT:    %Arg = alloca %class.MatrixClassTemplate, align 8
  // CHECK-NEXT:    store float* %Data, float** %Data.addr, align 8
  // CHECK-NEXT:    %0 = load float*, float** %Data.addr, align 8
  // CHECK-NEXT:    %1 = bitcast float* %0 to [150 x float]*
  // CHECK-NEXT:    %2 = bitcast [150 x float]* %1 to <150 x float>*
  // CHECK-NEXT:    %3 = load <150 x float>, <150 x float>* %2, align 4
  // CHECK-NEXT:    %Data1 = getelementptr inbounds %class.MatrixClassTemplate, %class.MatrixClassTemplate* %Arg, i32 0, i32 1
  // CHECK-NEXT:    %4 = bitcast [150 x float]* %Data1 to <150 x float>*
  // CHECK-NEXT:    store <150 x float> %3, <150 x float>* %4, align 4
  // CHECK-NEXT:    call void @_Z25matrix_template_referenceIfLj10ELj15EEvR19MatrixClassTemplateIT_XT0_EXT1_EES3_(%class.MatrixClassTemplate* nonnull align 8 dereferenceable(616) %Arg, %class.MatrixClassTemplate* nonnull align 8 dereferenceable(616) %agg.result)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr void @_Z25matrix_template_referenceIfLj10ELj15EEvR19MatrixClassTemplateIT_XT0_EXT1_EES3_(%class.MatrixClassTemplate* nonnull align 8 dereferenceable(616) %a, %class.MatrixClassTemplate* nonnull align 8 dereferenceable(616) %b)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca %class.MatrixClassTemplate*, align 8
  // CHECK-NEXT:    %b.addr = alloca %class.MatrixClassTemplate*, align 8
  // CHECK-NEXT:    store %class.MatrixClassTemplate* %a, %class.MatrixClassTemplate** %a.addr, align 8
  // CHECK-NEXT:    store %class.MatrixClassTemplate* %b, %class.MatrixClassTemplate** %b.addr, align 8
  // CHECK-NEXT:    %0 = load %class.MatrixClassTemplate*, %class.MatrixClassTemplate** %a.addr, align 8
  // CHECK-NEXT:    %Data = getelementptr inbounds %class.MatrixClassTemplate, %class.MatrixClassTemplate* %0, i32 0, i32 1
  // CHECK-NEXT:    %1 = bitcast [150 x float]* %Data to <150 x float>*
  // CHECK-NEXT:    %2 = load <150 x float>, <150 x float>* %1, align 4
  // CHECK-NEXT:    %3 = load %class.MatrixClassTemplate*, %class.MatrixClassTemplate** %b.addr, align 8
  // CHECK-NEXT:    %Data1 = getelementptr inbounds %class.MatrixClassTemplate, %class.MatrixClassTemplate* %3, i32 0, i32 1
  // CHECK-NEXT:    %4 = bitcast [150 x float]* %Data1 to <150 x float>*
  // CHECK-NEXT:    store <150 x float> %2, <150 x float>* %4, align 4
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

  // CHECK-LABEL: define void @_Z23test_template_deductionv()
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
  // CHECK-NEXT:    call void @_Z10use_matrixIiLm12EE8selectorILi3EERU11matrix_typeXLm10EEXT0_ET_([120 x i32]* nonnull align 4 dereferenceable(480) %m0)
  // CHECK-NEXT:    call void @_Z10use_matrixIiE8selectorILi2EERU11matrix_typeLm10ELm10ET_([100 x i32]* nonnull align 4 dereferenceable(400) %m1)
  // CHECK-NEXT:    call void @_Z10use_matrixIiLm12EE8selectorILi1EERU11matrix_typeXT0_EXLm10EET_([120 x i32]* nonnull align 4 dereferenceable(480) %m2)
  // CHECK-NEXT:    call void @_Z10use_matrixIiLm12ELm12EE8selectorILi0EERU11matrix_typeXT0_EXT1_ET_([144 x i32]* nonnull align 4 dereferenceable(576) %m3)
  // CHECK-NEXT:    call void @_Z10use_matrixILm12ELm12EE8selectorILi4EERU11matrix_typeXT_EXT0_Ef([144 x float]* nonnull align 4 dereferenceable(576) %m4)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiLm12EE8selectorILi3EERU11matrix_typeXLm10EEXT0_ET_([120 x i32]* nonnull align 4 dereferenceable(480) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [120 x i32]*, align 8
  // CHECK-NEXT:    store [120 x i32]* %m, [120 x i32]** %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiE8selectorILi2EERU11matrix_typeLm10ELm10ET_([100 x i32]* nonnull align 4 dereferenceable(400) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [100 x i32]*, align 8
  // CHECK-NEXT:    store [100 x i32]* %m, [100 x i32]** %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiLm12EE8selectorILi1EERU11matrix_typeXT0_EXLm10EET_([120 x i32]* nonnull align 4 dereferenceable(480) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [120 x i32]*, align 8
  // CHECK-NEXT:    store [120 x i32]* %m, [120 x i32]** %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixIiLm12ELm12EE8selectorILi0EERU11matrix_typeXT0_EXT1_ET_([144 x i32]* nonnull align 4 dereferenceable(576) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [144 x i32]*, align 8
  // CHECK-NEXT:    store [144 x i32]* %m, [144 x i32]** %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z10use_matrixILm12ELm12EE8selectorILi4EERU11matrix_typeXT_EXT0_Ef([144 x float]* nonnull align 4 dereferenceable(576)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [144 x float]*, align 8
  // CHECK-NEXT:    store [144 x float]* %m, [144 x float]** %m.addr, align 8
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
  // CHECK-LABEL: define void @_Z11test_auto_tv()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m = alloca [130 x i32], align 4
  // CHECK-NEXT:    call void @_Z3fooILm13EEvRU11matrix_typeXT_EXLm10EEi([130 x i32]* nonnull align 4 dereferenceable(520) %m)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr void @_Z3fooILm13EEvRU11matrix_typeXT_EXLm10EEi([130 x i32]* nonnull align 4 dereferenceable(520) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [130 x i32]*, align 8
  // CHECK-NEXT:    store [130 x i32]* %m, [130 x i32]** %m.addr, align 8
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
  // CHECK-LABEL: define void @_Z17test_use_matrix_2v()
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
  // CHECK-NEXT:    %call = call <40 x float> @_Z12use_matrix_2ILm4ELm6EEU11matrix_typeXplT_Li1EEXplT0_Li2EEfRU11matrix_typeXT_EXT0_Ei([24 x i32]* nonnull align 4 dereferenceable(96) %m1)
  // CHECK-NEXT:    %0 = bitcast [40 x float]* %r1 to <40 x float>*
  // CHECK-NEXT:    store <40 x float> %call, <40 x float>* %0, align 4
  // CHECK-NEXT:    call void @_Z12use_matrix_2ILm2ELm12EE8selectorILi0EERU11matrix_typeXplT_Li2EEXdvT0_Li2EEiRU11matrix_typeXT_EXT0_Ef([24 x i32]* nonnull align 4 dereferenceable(96) %m1, [24 x float]* nonnull align 4 dereferenceable(96) %m2)
  // CHECK-NEXT:    call void @_Z12use_matrix_2ILm5ELm8EE8selectorILi1EERU11matrix_typeXplT_T0_EXT0_EiRU11matrix_typeXT_EXmiT0_T_Ef([104 x i32]* nonnull align 4 dereferenceable(416) %m3, [15 x float]* nonnull align 4 dereferenceable(60) %m4)
  // CHECK-NEXT:    %call2 = call <20 x float> @_Z12use_matrix_2ILm5EEU11matrix_typeXplT_T_EXmiT_Li3EEfRU11matrix_typeXT_EXLm10EEi([50 x i32]* nonnull align 4 dereferenceable(200) %m5)
  // CHECK-NEXT:    %1 = bitcast [20 x float]* %r4 to <20 x float>*
  // CHECK-NEXT:    store <20 x float> %call2, <20 x float>* %1, align 4
  // CHECK-NEXT:    call void @_Z12use_matrix_3ILm6EE8selectorILi2EERU11matrix_typeXmiT_Li2EEXT_Ei([24 x i32]* nonnull align 4 dereferenceable(96) %m1)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define linkonce_odr <40 x float> @_Z12use_matrix_2ILm4ELm6EEU11matrix_typeXplT_Li1EEXplT0_Li2EEfRU11matrix_typeXT_EXT0_Ei([24 x i32]* nonnull align 4 dereferenceable(96) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [24 x i32]*, align 8
  // CHECK-NEXT:    store [24 x i32]* %m, [24 x i32]** %m.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z12use_matrix_2ILm2ELm12EE8selectorILi0EERU11matrix_typeXplT_Li2EEXdvT0_Li2EEiRU11matrix_typeXT_EXT0_Ef([24 x i32]* nonnull align 4 dereferenceable(96) %m1, [24 x float]* nonnull align 4 dereferenceable(96) %m2)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m1.addr = alloca [24 x i32]*, align 8
  // CHECK-NEXT:    %m2.addr = alloca [24 x float]*, align 8
  // CHECK-NEXT:    store [24 x i32]* %m1, [24 x i32]** %m1.addr, align 8
  // CHECK-NEXT:    store [24 x float]* %m2, [24 x float]** %m2.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z12use_matrix_2ILm5ELm8EE8selectorILi1EERU11matrix_typeXplT_T0_EXT0_EiRU11matrix_typeXT_EXmiT0_T_Ef([104 x i32]* nonnull align 4 dereferenceable(416) %m1, [15 x float]* nonnull align 4 dereferenceable(60) %m2)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m1.addr = alloca [104 x i32]*, align 8
  // CHECK-NEXT:    %m2.addr = alloca [15 x float]*, align 8
  // CHECK-NEXT:    store [104 x i32]* %m1, [104 x i32]** %m1.addr, align 8
  // CHECK-NEXT:    store [15 x float]* %m2, [15 x float]** %m2.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr <20 x float> @_Z12use_matrix_2ILm5EEU11matrix_typeXplT_T_EXmiT_Li3EEfRU11matrix_typeXT_EXLm10EEi([50 x i32]* nonnull align 4 dereferenceable(200) %m1)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m1.addr = alloca [50 x i32]*, align 8
  // CHECK-NEXT:    store [50 x i32]* %m1, [50 x i32]** %m1.addr, align 8
  // CHECK-NEXT:    call void @llvm.trap()
  // CHECK-NEXT:    unreachable

  // CHECK-LABEL: define linkonce_odr void @_Z12use_matrix_3ILm6EE8selectorILi2EERU11matrix_typeXmiT_Li2EEXT_Ei([24 x i32]* nonnull align 4 dereferenceable(96) %m)
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %m.addr = alloca [24 x i32]*, align 8
  // CHECK-NEXT:    store [24 x i32]* %m, [24 x i32]** %m.addr, align 8
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
