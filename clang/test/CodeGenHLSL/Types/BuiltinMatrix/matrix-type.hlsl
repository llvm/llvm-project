// RUN: %clang_cc1 -no-enable-noundef-analysis -triple spirv-unknown-vulkan-compute -fnative-half-type -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - -DSPIRV | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple dxil-pc-shadermodel6.3-compute -fnative-half-type -finclude-default-header %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: %struct.Matrix = type { i16, [12 x float], float }

// CHECK-LABEL:  define {{.*}}load_store_double
void load_store_double() {
  double4x4 a;
  double4x4 b;
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a = alloca [16 x double], align 8
  // CHECK-NEXT:    %b = alloca [16 x double], align 8
  // CHECK-NEXT:    [[tmp:%[0-9]*]] = load <16 x double>, ptr %b, align 8
  // CHECK-NEXT:    store <16 x double> [[tmp]], ptr %a, align 8
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_float
void load_store_float() {
  float3x4 a;
  float3x4 b;
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a = alloca [12 x float], align 4
  // CHECK-NEXT:    %b = alloca [12 x float], align 4
  // CHECK-NEXT:    [[tmp:%[0-9]*]] = load <12 x float>, ptr %b, align 4
  // CHECK-NEXT:    store <12 x float> [[tmp]], ptr %a, align 4
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_int
void load_store_int() {
  int3x4 a;
  int3x4 b;
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a = alloca [12 x i32], align 4
  // CHECK-NEXT:    %b = alloca [12 x i32], align 4
  // CHECK-NEXT:    [[tmp:%[0-9]*]] = load <12 x i32>, ptr %b, align 4
  // CHECK-NEXT:    store <12 x i32> [[tmp]], ptr %a, align 4
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_ull
void load_store_ull() {
  uint64_t3x4 a;
  uint64_t3x4 b;
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a = alloca [12 x i64], align 8
  // CHECK-NEXT:    %b = alloca [12 x i64], align 8
  // CHECK-NEXT:    [[tmp:%[0-9]*]] = load <12 x i64>, ptr %b, align 8
  // CHECK-NEXT:    store <12 x i64> [[tmp]], ptr %a, align 8
  // CHECK-NEXT:   ret void

  a = b;
}

// CHECK-LABEL:  define {{.*}}load_store_fp16
void load_store_fp16() {
  float16_t3x4 a;
  float16_t3x4 b;
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a = alloca [12 x half], align 2
  // CHECK-NEXT:    %b = alloca [12 x half], align 2
  // CHECK-NEXT:    [[tmp:%[0-9]*]] = load <12 x half>, ptr %b, align 2
  // CHECK-NEXT:    store <12 x half> [[tmp]], ptr %a, align 2
  // CHECK-NEXT:   ret void

  a = b;
}


typedef struct {
  uint16_t Tmp1;
  float3x4 Data;
  float Tmp2;
} Matrix;

// CHECK-LABEL: define {{.*}}matrix_struct
void matrix_struct() {
  Matrix a;
  Matrix b;
  // CHECK-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // CHECK-NEXT:    %a = alloca %struct.Matrix, align 4
  // CHECK-NEXT:    %b = alloca %struct.Matrix, align 4
  // CHECK-NEXT:    %Data = getelementptr inbounds nuw %struct.Matrix, ptr %a, i32 0, i32 1
  // CHECK-NEXT:    [[tmp:%[0-9]*]] = load <12 x float>, ptr %Data, align 4
  // CHECK-NEXT:    %Data1 = getelementptr inbounds nuw %struct.Matrix, ptr %b, i32 0, i32 1
  // CHECK-NEXT:    store <12 x float> [[tmp]], ptr %Data1, align 4
  // CHECK-NEXT:    ret void
  b.Data = a.Data;
}

// The following require matrix mangling, which is currenlty only available to SPIRV.
#ifdef SPIRV

// SPIRV-LABEL: define {{.*}}parameter_passing
void parameter_passing(in float3x3 a, inout float3x3 b, out float3x3 c) {
  // SPIRV-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // SPIRV-NEXT:    %a.addr = alloca [9 x float], align 4
  // SPIRV-NEXT:    %b.addr = alloca ptr, align 8
  // SPIRV-NEXT:    %c.addr = alloca ptr, align 8
  // SPIRV-NEXT:    store <9 x float> %a, ptr %a.addr, align 4
  // SPIRV-NEXT:    store ptr %b, ptr %b.addr, align 8
  // SPIRV-NEXT:    store ptr %c, ptr %c.addr, align 8
  // SPIRV-NEXT:    %1 = load <9 x float>, ptr %a.addr, align 4
  // SPIRV-NEXT:    %2 = load ptr, ptr %b.addr, align 8
  // SPIRV-NEXT:    store <9 x float> %1, ptr %2, align 4
  // SPIRV-NEXT:    %3 = load ptr, ptr %c.addr, align 8
  // SPIRV-NEXT:    store <9 x float> %1, ptr %3, align 4
  // SPIRV-NEXT:    ret void
  c = b = a;
}

// SPIRV-LABEL: define {{.*}}return_matrix
float3x3 return_matrix(inout float3x3 a) {
  // SPIRV-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // SPIRV-NEXT:    %a.addr = alloca ptr, align 8
  // SPIRV-NEXT:    store ptr %a, ptr %a.addr, align 8
  // SPIRV-NEXT:    %1 = load ptr, ptr %a.addr, align 8
  // SPIRV-NEXT:    %2 = load <9 x float>, ptr %1, align 4
  // SPIRV-NEXT:    ret <9 x float> %2
  return a;
}


class MatrixClass {
  int Tmp1;
  float3x4 Data;
  int64_t Tmp2;
};

// SPIRV-LABEL: define {{.*}}matrix_class_reference
void matrix_class_reference(inout MatrixClass a, inout MatrixClass b) {
  // SPIRV-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // SPIRV-NEXT:    %a.addr = alloca ptr, align 8
  // SPIRV-NEXT:    %b.addr = alloca ptr, align 8
  // SPIRV-NEXT:    store ptr %a, ptr %a.addr, align 8
  // SPIRV-NEXT:    store ptr %b, ptr %b.addr, align 8
  // SPIRV-NEXT:    %1 = load ptr, ptr %a.addr, align 8
  // SPIRV-NEXT:    %Data = getelementptr inbounds nuw %class.MatrixClass, ptr %1, i32 0, i32 1
  // SPIRV-NEXT:    %2 = load <12 x float>, ptr %Data, align 4
  // SPIRV-NEXT:    %3 = load ptr, ptr %b.addr, align 8
  // SPIRV-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClass, ptr %3, i32 0, i32 1
  // SPIRV-NEXT:    store <12 x float> %2, ptr %Data1, align 4
  // SPIRV-NEXT:    ret void
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

// SPIRV-LABEL: define {{.*}}matrix_template_reference_caller
MatrixClassTemplate<float, 3, 4> matrix_template_reference_caller(matrix<float,3,4> Data) {
  // SPIRV-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // SPIRV-NEXT:    %Data.addr = alloca [12 x float], align 4
  // SPIRV-NEXT:    %Arg = alloca %class.MatrixClassTemplate, align 8
  // SPIRV-NEXT:    %tmp = alloca %class.MatrixClassTemplate, align 8
  // SPIRV-NEXT:    %tmp2 = alloca %class.MatrixClassTemplate, align 8
  // SPIRV-NEXT:    store <12 x float> %Data, ptr %Data.addr, align 4
  // SPIRV-NEXT:    %1 = load <12 x float>, ptr %Data.addr, align 4
  // SPIRV-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr %Arg, i32 0, i32 1
  // SPIRV-NEXT:    store <12 x float> %1, ptr %Data1, align 4
  // SPIRV-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp, ptr align 8 %Arg, i64 64, i1 false)
  // SPIRV-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp2, ptr align 8 %agg.result, i64 64, i1 false)
  // SPIRV-NEXT:    call{{.*}} void @_Z25matrix_template_referenceIfLj3ELj4EEv19MatrixClassTemplateIT_XT0_EXT1_EES2_(ptr noalias nonnull align 8 dereferenceable(64) %tmp, ptr noalias nonnull align 8 dereferenceable(64) %tmp2)
  // SPIRV-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %Arg, ptr align 8 %tmp, i64 64, i1 false)
  // SPIRV-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %tmp2, i64 64, i1 false)
  // SPIRV-NEXT:    ret void

  // SPIRV-LABEL: define{{.*}} void @_Z25matrix_template_referenceIfLj3ELj4EEv19MatrixClassTemplateIT_XT0_EXT1_EES2_(ptr noalias nonnull align 8 dereferenceable(64) %a, ptr noalias nonnull align 8 dereferenceable(64) %b)
  // SPIRV-NEXT:  entry:
  // SPIRV-NEXT:    %0 = call token @llvm.experimental.convergence.entry()
  // SPIRV-NEXT:    %a.addr = alloca ptr, align 8
  // SPIRV-NEXT:    %b.addr = alloca ptr, align 8
  // SPIRV-NEXT:    store ptr %a, ptr %a.addr, align 8
  // SPIRV-NEXT:    store ptr %b, ptr %b.addr, align 8
  // SPIRV-NEXT:    %1 = load ptr, ptr %a.addr, align 8
  // SPIRV-NEXT:    %Data = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr %1, i32 0, i32 1
  // SPIRV-NEXT:    %2 = load <12 x float>, ptr %Data, align 4
  // SPIRV-NEXT:    %3 = load ptr, ptr %b.addr, align 8
  // SPIRV-NEXT:    %Data1 = getelementptr inbounds nuw %class.MatrixClassTemplate, ptr %3, i32 0, i32 1
  // SPIRV-NEXT:    store <12 x float> %2, ptr %Data1, align 4
  // SPIRV-NEXT:    ret void

  MatrixClassTemplate<float, 3, 4> Result, Arg;
  Arg.Data = Data;
  matrix_template_reference(Arg, Result);
  return Result;
}

#endif

