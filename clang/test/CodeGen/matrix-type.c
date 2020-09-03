// RUN: %clang_cc1 -fenable-matrix -triple x86_64-apple-darwin %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

#if !__has_extension(matrix_types)
#error Expected extension 'matrix_types' to be enabled
#endif

typedef double dx5x5_t __attribute__((matrix_type(5, 5)));

// CHECK: %struct.Matrix = type { i8, [12 x float], float }

void load_store_double(dx5x5_t *a, dx5x5_t *b) {
  // CHECK-LABEL:  define void @load_store_double(
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

typedef float fx3x4_t __attribute__((matrix_type(3, 4)));
void load_store_float(fx3x4_t *a, fx3x4_t *b) {
  // CHECK-LABEL:  define void @load_store_float(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [12 x float]*, align 8
  // CHECK-NEXT:    %b.addr = alloca [12 x float]*, align 8
  // CHECK-NEXT:    store [12 x float]* %a, [12 x float]** %a.addr, align 8
  // CHECK-NEXT:    store [12 x float]* %b, [12 x float]** %b.addr, align 8
  // CHECK-NEXT:    %0 = load [12 x float]*, [12 x float]** %b.addr, align 8
  // CHECK-NEXT:    %1 = bitcast [12 x float]* %0 to <12 x float>*
  // CHECK-NEXT:    %2 = load <12 x float>, <12 x float>* %1, align 4
  // CHECK-NEXT:    %3 = load [12 x float]*, [12 x float]** %a.addr, align 8
  // CHECK-NEXT:    %4 = bitcast [12 x float]* %3 to <12 x float>*
  // CHECK-NEXT:    store <12 x float> %2, <12 x float>* %4, align 4
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef int ix3x4_t __attribute__((matrix_type(4, 3)));
void load_store_int(ix3x4_t *a, ix3x4_t *b) {
  // CHECK-LABEL:  define void @load_store_int(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [12 x i32]*, align 8
  // CHECK-NEXT:    %b.addr = alloca [12 x i32]*, align 8
  // CHECK-NEXT:    store [12 x i32]* %a, [12 x i32]** %a.addr, align 8
  // CHECK-NEXT:    store [12 x i32]* %b, [12 x i32]** %b.addr, align 8
  // CHECK-NEXT:    %0 = load [12 x i32]*, [12 x i32]** %b.addr, align 8
  // CHECK-NEXT:    %1 = bitcast [12 x i32]* %0 to <12 x i32>*
  // CHECK-NEXT:    %2 = load <12 x i32>, <12 x i32>* %1, align 4
  // CHECK-NEXT:    %3 = load [12 x i32]*, [12 x i32]** %a.addr, align 8
  // CHECK-NEXT:    %4 = bitcast [12 x i32]* %3 to <12 x i32>*
  // CHECK-NEXT:    store <12 x i32> %2, <12 x i32>* %4, align 4
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef unsigned long long ullx3x4_t __attribute__((matrix_type(4, 3)));
void load_store_ull(ullx3x4_t *a, ullx3x4_t *b) {
  // CHECK-LABEL:  define void @load_store_ull(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [12 x i64]*, align 8
  // CHECK-NEXT:    %b.addr = alloca [12 x i64]*, align 8
  // CHECK-NEXT:    store [12 x i64]* %a, [12 x i64]** %a.addr, align 8
  // CHECK-NEXT:    store [12 x i64]* %b, [12 x i64]** %b.addr, align 8
  // CHECK-NEXT:    %0 = load [12 x i64]*, [12 x i64]** %b.addr, align 8
  // CHECK-NEXT:    %1 = bitcast [12 x i64]* %0 to <12 x i64>*
  // CHECK-NEXT:    %2 = load <12 x i64>, <12 x i64>* %1, align 8
  // CHECK-NEXT:    %3 = load [12 x i64]*, [12 x i64]** %a.addr, align 8
  // CHECK-NEXT:    %4 = bitcast [12 x i64]* %3 to <12 x i64>*
  // CHECK-NEXT:    store <12 x i64> %2, <12 x i64>* %4, align 8
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef __fp16 fp16x3x4_t __attribute__((matrix_type(4, 3)));
void load_store_fp16(fp16x3x4_t *a, fp16x3x4_t *b) {
  // CHECK-LABEL:  define void @load_store_fp16(
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [12 x half]*, align 8
  // CHECK-NEXT:    %b.addr = alloca [12 x half]*, align 8
  // CHECK-NEXT:    store [12 x half]* %a, [12 x half]** %a.addr, align 8
  // CHECK-NEXT:    store [12 x half]* %b, [12 x half]** %b.addr, align 8
  // CHECK-NEXT:    %0 = load [12 x half]*, [12 x half]** %b.addr, align 8
  // CHECK-NEXT:    %1 = bitcast [12 x half]* %0 to <12 x half>*
  // CHECK-NEXT:    %2 = load <12 x half>, <12 x half>* %1, align 2
  // CHECK-NEXT:    %3 = load [12 x half]*, [12 x half]** %a.addr, align 8
  // CHECK-NEXT:    %4 = bitcast [12 x half]* %3 to <12 x half>*
  // CHECK-NEXT:    store <12 x half> %2, <12 x half>* %4, align 2
  // CHECK-NEXT:   ret void

  *a = *b;
}

typedef float fx3x3_t __attribute__((matrix_type(3, 3)));

void parameter_passing(fx3x3_t a, fx3x3_t *b) {
  // CHECK-LABEL: define void @parameter_passing(
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
  // CHECK-LABEL: define <9 x float> @return_matrix
  // CHECK-NEXT:  entry:
  // CHECK-NEXT:    %a.addr = alloca [9 x float]*, align 8
  // CHECK-NEXT:    store [9 x float]* %a, [9 x float]** %a.addr, align 8
  // CHECK-NEXT:    %0 = load [9 x float]*, [9 x float]** %a.addr, align 8
  // CHECK-NEXT:    %1 = bitcast [9 x float]* %0 to <9 x float>*
  // CHECK-NEXT:    %2 = load <9 x float>, <9 x float>* %1, align 4
  // CHECK-NEXT:    ret <9 x float> %2
  return *a;
}

typedef struct {
  char Tmp1;
  fx3x4_t Data;
  float Tmp2;
} Matrix;

void matrix_struct(Matrix *a, Matrix *b) {
  // CHECK-LABEL: define void @matrix_struct(
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
