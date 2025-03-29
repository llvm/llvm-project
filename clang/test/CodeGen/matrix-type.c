// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

#if !__has_extension(matrix_types)
#error Expected extension 'matrix_types' to be enabled
#endif

#if !__has_extension(matrix_types_scalar_division)
#error Expected extension 'matrix_types_scalar_division' to be enabled
#endif

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));

// CHECK: %struct.Matrix = type { i8, [12 x float], float }

void load_store_double(dx5x5_t *a, dx5x5_t *b) {
  // CHECK-LABEL:  define{{.*}} void @load_store_double(
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

typedef float fx3x4_t __attribute__((matrix_type(3, 4)));
void load_store_float(fx3x4_t *a, fx3x4_t *b) {
  // CHECK-LABEL:  define{{.*}} void @load_store_float(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %1 = load <12 x float>, ptr %0, align 4
  // CHECK-NEXT:    %2 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    store <12 x float> %1, ptr %2, align 4
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef int ix3x4_t __attribute__((matrix_type(4, 3)));
void load_store_int(ix3x4_t *a, ix3x4_t *b) {
  // CHECK-LABEL:  define{{.*}} void @load_store_int(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %1 = load <12 x i32>, ptr %0, align 4
  // CHECK-NEXT:    %2 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    store <12 x i32> %1, ptr %2, align 4
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef unsigned long long ullx3x4_t __attribute__((matrix_type(4, 3)));
void load_store_ull(ullx3x4_t *a, ullx3x4_t *b) {
  // CHECK-LABEL:  define{{.*}} void @load_store_ull(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %1 = load <12 x i64>, ptr %0, align 8
  // CHECK-NEXT:    %2 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    store <12 x i64> %1, ptr %2, align 8
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef __fp16 fp16x3x4_t __attribute__((matrix_type(4, 3)));
void load_store_fp16(fp16x3x4_t *a, fp16x3x4_t *b) {
  // CHECK-LABEL:  define{{.*}} void @load_store_fp16(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    %b.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %b.addr, align 8
  // CHECK-NEXT:    %1 = load <12 x half>, ptr %0, align 2
  // CHECK-NEXT:    %2 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    store <12 x half> %1, ptr %2, align 2
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef float fx3x3_t __attribute__((matrix_type(3, 3)));

void parameter_passing(fx3x3_t a, fx3x3_t *b) {
  // CHECK-LABEL: define{{.*}} void @parameter_passing(
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
  // CHECK-LABEL: define{{.*}} <9 x float> @return_matrix
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca ptr, align 8
  // CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
  // CHECK-NEXT:    %0 = load ptr, ptr %a.addr, align 8
  // CHECK-NEXT:    %1 = load <9 x float>, ptr %0, align 4
  // CHECK-NEXT:    ret <9 x float> %1
  return *a;
}

typedef struct {
  char Tmp1;
  fx3x4_t Data;
  float Tmp2;
} Matrix;

void matrix_struct(Matrix *a, Matrix *b) {
  // CHECK-LABEL: define{{.*}} void @matrix_struct(
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

typedef double dx4x4_t __attribute__((matrix_type(4, 4)));
void matrix_inline_asm_memory_readwrite(void) {
  // CHECK-LABEL: define{{.*}} void @matrix_inline_asm_memory_readwrite()
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    [[ALLOCA:%.+]] = alloca [16 x double], align 8
  // CHECK-NEXT:    [[VAL:%.+]] = load <16 x double>, ptr [[ALLOCA]], align 8
  // CHECK-NEXT:    call void asm sideeffect "", "=*r|m,0,~{memory},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(<16 x double>) [[ALLOCA]], <16 x double> [[VAL]])
  // CHECK-NEXT:    ret void

  dx4x4_t m;
  asm volatile(""
               : "+r,m"(m)
               :
               : "memory");
}
