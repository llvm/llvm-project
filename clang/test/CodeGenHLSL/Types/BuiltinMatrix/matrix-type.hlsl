// RUN: %clang_cc1 -no-enable-noundef-analysis -triple spirv-unknown-vulkan-compute -fnative-half-type -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple dxil-pc-shadermodel6.3-compute -fnative-half-type -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %struct.Matrix = type { i16, [12 x float], float }

// CHECK-LABEL:  define {{.*}}load_store_double
void load_store_double(inout double4x4 a,  inout double4x4 b) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    [[B_PTR:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    [[B:%.*]] = load <16 x double>, ptr [[B_PTR]], align 8
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    store <16 x double> [[B]], ptr [[A_PTR]], align 8
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_float
void load_store_float(inout float3x4 a,  inout float3x4 b) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    [[B_PTR:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    [[B:%.*]] = load <12 x float>, ptr [[B_PTR]], align 4
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    store <12 x float> [[B]], ptr [[A_PTR]], align 4
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_int
void load_store_int(inout int3x4 a, inout int3x4 b) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    [[B_PTR:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    [[B:%.*]] = load <12 x i32>, ptr [[B_PTR]], align 4
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    store <12 x i32> [[B]], ptr [[A_PTR]], align 4
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_ull
void load_store_ull(inout uint64_t3x4 a,  inout uint64_t3x4 b) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    [[B_PTR:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    [[B:%.*]] = load <12 x i64>, ptr [[B_PTR]], align 8
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    store <12 x i64> [[B]], ptr [[A_PTR]], align 8
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_fp16
void load_store_fp16(inout float16_t3x4 a, inout float16_t3x4 b) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    [[B_PTR:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    [[B:%.*]] = load <12 x half>, ptr [[B_PTR]], align 2
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    store <12 x half> [[B]], ptr [[A_PTR]], align 2
  // CHECK-NEXT:   ret void

  a = b;
}


typedef struct {
  uint16_t Tmp1;
  float3x4 Data;
  float Tmp2;
} Matrix;

// CHECK-LABEL: define {{.*}}matrix_struct
void matrix_struct(Matrix a,  Matrix b) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %struct.Matrix, ptr %a, i32 0, i32 1
  // CHECK-NEXT:    [[tmp:%[0-9]*]] = load <12 x float>, ptr %Data, align 4
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %struct.Matrix, ptr %b, i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> [[tmp]], ptr %Data1, align 4
  // CHECK-NEXT:    ret void
  b.Data = a.Data;
}

// CHECK-LABEL: define {{.*}}parameter_passing
void parameter_passing(in float3x3 a, inout float3x3 b, out float3x3 c) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca [9 x float], align 4
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    %c.addr = alloca ptr, align
  // CHECK-NEXT:    store <9 x float> %a, ptr %a.addr, align 4
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    store ptr %c, ptr %c.addr, align
  // CHECK-NEXT:    [[A:%.*]] = load <9 x float>, ptr %a.addr, align 4
  // CHECK-NEXT:    [[B:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    store <9 x float> [[A]], ptr [[B]], align 4
  // CHECK-NEXT:    [[C:%.*]] = load ptr, ptr %c.addr, align
  // CHECK-NEXT:    store <9 x float> [[A]], ptr [[C]], align 4
  // CHECK-NEXT:    ret void
  c = b = a;
}

// CHECK-LABEL: define {{.*}}return_matrix
float3x3 return_matrix(inout float3x3 a) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    [[A:%.*]] = load <9 x float>, ptr [[A_PTR]], align 4
  // CHECK-NEXT:    ret <9 x float> [[A]]
  return a;
}


class MatrixClass {
  int Tmp1;
  float3x4 Data;
  int64_t Tmp2;
};

// CHECK-LABEL: define {{.*}}matrix_class_reference
void matrix_class_reference(inout MatrixClass a, inout MatrixClass b) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %class.MatrixClass, ptr [[A_PTR]], i32 0, i32 1
  // CHECK-NEXT:    [[DATA:%.*]] = load <12 x float>, ptr %Data, align 4
  // CHECK-NEXT:    [[B_PTR:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClass, ptr [[B_PTR]], i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> [[DATA]], ptr %Data1, align 4
  // CHECK-NEXT:    ret void
  b.Data = a.Data;
}

template <typename Ty, unsigned Rows, unsigned Cols>
class MatrixClassTemplate {
  using MatrixTy = matrix<Ty, Rows, Cols>;
  int Tmp1;
  MatrixTy Data;
  int64_t Tmp2;
};

template <typename Ty, unsigned Rows, unsigned Cols>
void matrix_template_reference(inout MatrixClassTemplate<Ty, Rows, Cols> a, inout MatrixClassTemplate<Ty, Rows, Cols> b) {
  b.Data = a.Data;
}

// CHECK-LABEL: define {{.*}}matrix_template_reference_caller
MatrixClassTemplate<float, 3, 4> matrix_template_reference_caller(matrix<float,3,4> Data) {
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %Data.addr = alloca [12 x float], align 4
  // CHECK-NEXT:    %Arg = alloca %class.MatrixClassTemplate, align 8
  // CHECK-NEXT:    %tmp = alloca %class.MatrixClassTemplate, align 8
  // CHECK-NEXT:    %tmp2 = alloca %class.MatrixClassTemplate, align 8
  // CHECK-NEXT:    store <12 x float> %Data, ptr %Data.addr, align 4
  // CHECK-NEXT:    [[DATA:%.*]] = load <12 x float>, ptr %Data.addr, align 4
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr %Arg, i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> [[DATA]], ptr %Data1, align 4
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i{{[0-9]*}}(ptr align 8 %tmp, ptr align 8 %Arg, i{{[0-9]*}} 64, i1 false)
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i{{[0-9]*}}(ptr align 8 %tmp2, ptr align 8 %agg.result, i{{[0-9]*}} 64, i1 false)
  // CHECK-NEXT:    call{{.*}} void @_Z25matrix_template_referenceIfLj3ELj4EEv19MatrixClassTemplateIT_XT0_EXT1_EES2_(ptr noalias nonnull align 8 dereferenceable(64) %tmp, ptr noalias nonnull align 8 dereferenceable(64) %tmp2)
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i{{[0-9]*}}(ptr align 8 %Arg, ptr align 8 %tmp, i{{[0-9]*}} 64, i1 false)
  // CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i{{[0-9]*}}(ptr align 8 %agg.result, ptr align 8 %tmp2, i{{[0-9]*}} 64, i1 false)
  // CHECK-NEXT:    ret void

  // CHECK-LABEL: define{{.*}} void @_Z25matrix_template_referenceIfLj3ELj4EEv19MatrixClassTemplateIT_XT0_EXT1_EES2_(ptr noalias nonnull align 8 dereferenceable(64) %a, ptr noalias nonnull align 8 dereferenceable(64) %b)
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a.addr = alloca ptr, align
  // CHECK-NEXT:    %b.addr = alloca ptr, align
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align
  // CHECK-NEXT:    [[A_PTR:%.*]] = load ptr, ptr %a.addr, align
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr [[A_PTR]], i32 0, i32 1
  // CHECK-NEXT:    [[DATA:%.*]] = load <12 x float>, ptr %Data, align 4
  // CHECK-NEXT:    [[B_PTR:%.*]] = load ptr, ptr %b.addr, align
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr [[B_PTR]], i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> [[DATA]], ptr %Data1, align 4
  // CHECK-NEXT:    ret void

  MatrixClassTemplate<float, 3, 4> Result, Arg;
  Arg.Data = Data;
  matrix_template_reference(Arg, Result);
  return Result;
}


