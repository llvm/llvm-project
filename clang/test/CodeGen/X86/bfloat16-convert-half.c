// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -fenable-matrix -emit-llvm \
// RUN:   %s -o - | opt -S -passes=mem2reg | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -fenable-matrix -emit-llvm \
// RUN:   -target-feature +avx512fp16 %s -o - | opt -S -passes=mem2reg | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -fenable-matrix -emit-llvm \
// RUN:   -fnative-half-type %s -o - | opt -S -passes=mem2reg | FileCheck %s

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef __fp16 fp16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x2x2 __attribute__((matrix_type(2, 2)));
typedef _Float16 f16x2x2 __attribute__((matrix_type(2, 2)));

// CHECK-LABEL: define dso_local half @test_convert_from_bf16_to_fp16(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    ret half [[FPTRUNC]]
//
_Float16 test_convert_from_bf16_to_fp16(__bf16 a) {
    return (_Float16)a;
}

// CHECK-LABEL: define dso_local bfloat @test_convert_from_fp16_to_bf16(
// CHECK-SAME: half noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext half [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to bfloat
// CHECK-NEXT:    ret bfloat [[FPTRUNC]]
//
__bf16 test_convert_from_fp16_to_bf16(_Float16 a) {
    return (__bf16)a;
}

// GH170434
// CHECK-LABEL: define dso_local bfloat @test_fmaxf16_bf16(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext nsz bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc nsz float [[FPEXT]] to half
// CHECK-NEXT:    [[FPEXT1:%.*]] = fpext nsz bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC2:%.*]] = fptrunc nsz float [[FPEXT1]] to half
// CHECK-NEXT:    [[TMP0:%.*]] = call nsz half @llvm.maxnum.f16(half [[FPTRUNC]], half [[FPTRUNC2]])
// CHECK-NEXT:    [[FPEXT3:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    [[FPTRUNC4:%.*]] = fptrunc float [[FPEXT3]] to bfloat
// CHECK-NEXT:    ret bfloat [[FPTRUNC4]]
//
__bf16 test_fmaxf16_bf16(__bf16 a) {
  return __builtin_fmaxf16(a, a);
}

// CHECK-LABEL: define dso_local half @test_return_bf16_as_f16(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    ret half [[FPTRUNC]]
//
_Float16 test_return_bf16_as_f16(__bf16 a) {
  return a;
}

void take_f16(_Float16);
// CHECK-LABEL: define dso_local void @test_pass_bf16_as_f16(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    call void @take_f16(half noundef [[FPTRUNC]])
// CHECK-NEXT:    ret void
//
void test_pass_bf16_as_f16(__bf16 a) {
  take_f16(a);
}

// CHECK-LABEL: define dso_local void @test_assign_f16_to_bf16(
// CHECK-SAME: ptr noundef [[P:%.*]], half noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext half [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to bfloat
// CHECK-NEXT:    store bfloat [[FPTRUNC]], ptr [[P]], align 2
// CHECK-NEXT:    ret void
//
void test_assign_f16_to_bf16(__bf16 *p, _Float16 a) {
  *p = a;
}

// CHECK-LABEL: define dso_local void @test_assign_bf16_to_fp16(
// CHECK-SAME: ptr noundef [[P:%.*]], bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    store half [[FPTRUNC]], ptr [[P]], align 2
// CHECK-NEXT:    ret void
//
void test_assign_bf16_to_fp16(__fp16 *p, __bf16 a) {
  *p = a;
}

// CHECK-LABEL: define dso_local bfloat @test_return_fp16_as_bf16(
// CHECK-SAME: ptr noundef [[P:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, ptr [[P]], align 2
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to bfloat
// CHECK-NEXT:    ret bfloat [[FPTRUNC]]
//
__bf16 test_return_fp16_as_bf16(__fp16 *p) {
  return *p;
}

// CHECK-LABEL: define dso_local void @test_cast_bf16_to_fp16(
// CHECK-SAME: ptr noundef [[P:%.*]], bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    store half [[FPTRUNC]], ptr [[P]], align 2
// CHECK-NEXT:    ret void
//
void test_cast_bf16_to_fp16(__fp16 *p, __bf16 a) {
  *p = (__fp16)a;
}

// CHECK-LABEL: define dso_local bfloat @test_cast_fp16_to_bf16(
// CHECK-SAME: ptr noundef [[P:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, ptr [[P]], align 2
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to bfloat
// CHECK-NEXT:    ret bfloat [[FPTRUNC]]
//
__bf16 test_cast_fp16_to_bf16(__fp16 *p) {
  return (__bf16)*p;
}

// CHECK-LABEL: define dso_local void @test_compound_assign_fp16_bf16(
// CHECK-SAME: ptr noundef [[P:%.*]], bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    [[TMP0:%.*]] = load half, ptr [[P]], align 2
// CHECK-NEXT:    [[ADD:%.*]] = fadd half [[TMP0]], [[FPTRUNC]]
// CHECK-NEXT:    store half [[ADD]], ptr [[P]], align 2
// CHECK-NEXT:    ret void
//
void test_compound_assign_fp16_bf16(__fp16 *p, __bf16 a) {
  *p += a;
}

// CHECK-LABEL: define dso_local <8 x half> @test_convertvector_bf16_to_f16(
// CHECK-SAME: <8 x bfloat> noundef [[V:%.*]]) #[[ATTR3:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <8 x bfloat> [[V]] to <8 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <8 x float> [[FPEXT]] to <8 x half>
// CHECK-NEXT:    ret <8 x half> [[FPTRUNC]]
//
f16x8 test_convertvector_bf16_to_f16(bf16x8 v) {
  return __builtin_convertvector(v, f16x8);
}

// CHECK-LABEL: define dso_local <8 x bfloat> @test_convertvector_f16_to_bf16(
// CHECK-SAME: <8 x half> noundef [[V:%.*]]) #[[ATTR3]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <8 x half> [[V]] to <8 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <8 x float> [[FPEXT]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[FPTRUNC]]
//
bf16x8 test_convertvector_f16_to_bf16(f16x8 v) {
  return __builtin_convertvector(v, bf16x8);
}

// CHECK-LABEL: define dso_local <8 x half> @test_convertvector_bf16_to_fp16(
// CHECK-SAME: <8 x bfloat> noundef [[V:%.*]]) #[[ATTR3]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <8 x bfloat> [[V]] to <8 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <8 x float> [[FPEXT]] to <8 x half>
// CHECK-NEXT:    ret <8 x half> [[FPTRUNC]]
//
fp16x8 test_convertvector_bf16_to_fp16(bf16x8 v) {
  return __builtin_convertvector(v, fp16x8);
}

// CHECK-LABEL: define dso_local void @test_matrix_cast_bf16_to_f16(
// CHECK-SAME: ptr noundef [[IN:%.*]], ptr noundef [[OUT:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x bfloat>, ptr [[IN]], align 2
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <4 x bfloat> [[TMP0]] to <4 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <4 x float> [[FPEXT]] to <4 x half>
// CHECK-NEXT:    store <4 x half> [[FPTRUNC]], ptr [[OUT]], align 2
// CHECK-NEXT:    ret void
//
void test_matrix_cast_bf16_to_f16(bf16x2x2 *in, f16x2x2 *out) {
  *out = (f16x2x2)*in;
}

// CHECK-LABEL: define dso_local void @test_matrix_cast_f16_to_bf16(
// CHECK-SAME: ptr noundef [[IN:%.*]], ptr noundef [[OUT:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x half>, ptr [[IN]], align 2
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <4 x half> [[TMP0]] to <4 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <4 x float> [[FPEXT]] to <4 x bfloat>
// CHECK-NEXT:    store <4 x bfloat> [[FPTRUNC]], ptr [[OUT]], align 2
// CHECK-NEXT:    ret void
//
void test_matrix_cast_f16_to_bf16(f16x2x2 *in, bf16x2x2 *out) {
  *out = (bf16x2x2)*in;
}

// CHECK-LABEL: define dso_local half @test_reduce_fadd_bf16_start(
// CHECK-SAME: <8 x half> noundef [[V:%.*]], bfloat noundef [[S:%.*]]) #[[ATTR3]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[S]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    [[RDX_ADDF:%.*]] = call half @llvm.vector.reduce.fadd.v8f16(half [[FPTRUNC]], <8 x half> [[V]])
// CHECK-NEXT:    ret half [[RDX_ADDF]]
//
_Float16 test_reduce_fadd_bf16_start(f16x8 v, __bf16 s) {
  return __builtin_reduce_in_order_fadd(v, s);
}

// CHECK-LABEL: define dso_local float @test_bf16_to_float(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    ret float [[CONV]]
//
float test_bf16_to_float(__bf16 a) {
  return a;
}

// CHECK-LABEL: define dso_local bfloat @test_float_to_bf16(
// CHECK-SAME: float noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fptrunc float [[A]] to bfloat
// CHECK-NEXT:    ret bfloat [[CONV]]
//
__bf16 test_float_to_bf16(float a) {
  return a;
}

// CHECK-LABEL: define dso_local double @test_bf16_to_double(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fpext bfloat [[A]] to double
// CHECK-NEXT:    ret double [[CONV]]
//
double test_bf16_to_double(__bf16 a) {
  return a;
}

// CHECK-LABEL: define dso_local bfloat @test_double_to_bf16(
// CHECK-SAME: double noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fptrunc double [[A]] to bfloat
// CHECK-NEXT:    ret bfloat [[CONV]]
//
__bf16 test_double_to_bf16(double a) {
  return a;
}

// CHECK-LABEL: define dso_local float @test_f16_to_float(
// CHECK-SAME: half noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fpext half [[A]] to float
// CHECK-NEXT:    ret float [[CONV]]
//
float test_f16_to_float(_Float16 a) {
  return a;
}

// CHECK-LABEL: define dso_local half @test_float_to_f16(
// CHECK-SAME: float noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fptrunc float [[A]] to half
// CHECK-NEXT:    ret half [[CONV]]
//
_Float16 test_float_to_f16(float a) {
  return a;
}

// CHECK-LABEL: define dso_local float @test_fp16_to_float(
// CHECK-SAME: ptr noundef [[P:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, ptr [[P]], align 2
// CHECK-NEXT:    [[CONV:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    ret float [[CONV]]
//
float test_fp16_to_float(__fp16 *p) {
  return *p;
}

// CHECK-LABEL: define dso_local void @test_float_to_fp16(
// CHECK-SAME: ptr noundef [[P:%.*]], float noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fptrunc float [[A]] to half
// CHECK-NEXT:    store half [[CONV]], ptr [[P]], align 2
// CHECK-NEXT:    ret void
//
void test_float_to_fp16(__fp16 *p, float a) {
  *p = a;
}

// CHECK-LABEL: define dso_local <8 x float> @test_convertvector_bf16_to_float(
// CHECK-SAME: <8 x bfloat> noundef [[V:%.*]]) #[[ATTR4:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CONV:%.*]] = fpext <8 x bfloat> [[V]] to <8 x float>
// CHECK-NEXT:    ret <8 x float> [[CONV]]
//
f32x8 test_convertvector_bf16_to_float(bf16x8 v) {
  return __builtin_convertvector(v, f32x8);
}
