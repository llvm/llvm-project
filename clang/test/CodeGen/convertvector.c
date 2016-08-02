// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -target-cpu corei7-avx -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -target-cpu corei7-avx -emit-llvm -x c++ %s -o - | FileCheck %s

typedef double vector8double __attribute__((__vector_size__(64)));
typedef float  vector8float  __attribute__((__vector_size__(32)));
typedef long   vector8long   __attribute__((__vector_size__(64)));
typedef short  vector8short  __attribute__((__vector_size__(16)));
typedef unsigned long   vector8ulong   __attribute__((__vector_size__(64)));
typedef unsigned short  vector8ushort  __attribute__((__vector_size__(16)));

#ifdef __cplusplus
#define BOOL bool
#else
#define BOOL _Bool
#endif

typedef BOOL vector8bool __attribute__((__ext_vector_type__(8)));

#ifdef __cplusplus
extern "C" {
#endif

vector8float flt_trunc(vector8double x) {
  return __builtin_convertvector(x, vector8float);
  // CHECK-LABEL: @flt_trunc
  // CHECK: fptrunc <8 x double> %{{[^ ]}} to <8 x float>
}

vector8double flt_ext(vector8float x) {
  return __builtin_convertvector(x, vector8double);
  // CHECK-LABEL: @flt_ext
  // CHECK: fpext <8 x float> %{{[^ ]}} to <8 x double>
}

vector8bool flt_tobool(vector8float x) {
  return __builtin_convertvector(x, vector8bool);
  // CHECK-LABEL: @flt_tobool
  // CHECK-NOT: fptoui <8 x float> %{{[^ ]}} to <8 x i1>
  // CHECK: fcmp une <8 x float> %{{[^ ]}}, zeroinitializer
}

vector8long flt_tosi(vector8float x) {
  return __builtin_convertvector(x, vector8long);
  // CHECK-LABEL: @flt_tosi
  // CHECK: fptosi <8 x float> %{{[^ ]}} to <8 x i64>
}

vector8ulong flt_toui(vector8float x) {
  return __builtin_convertvector(x, vector8ulong);
  // CHECK-LABEL: @flt_toui
  // CHECK: fptoui <8 x float> %{{[^ ]}} to <8 x i64>
}

vector8ulong fltd_toui(vector8double x) {
  return __builtin_convertvector(x, vector8ulong);
  // CHECK-LABEL: @fltd_toui
  // CHECK: fptoui <8 x double> %{{[^ ]}} to <8 x i64>
}

vector8ulong int_zext(vector8ushort x) {
  return __builtin_convertvector(x, vector8ulong);
  // CHECK-LABEL: @int_zext
  // CHECK: zext <8 x i16> %{{[^ ]}} to <8 x i64>
}

vector8long int_sext(vector8short x) {
  return __builtin_convertvector(x, vector8long);
  // CHECK-LABEL: @int_sext
  // CHECK: sext <8 x i16> %{{[^ ]}} to <8 x i64>
}

vector8bool int_tobool(vector8short x) {
  return __builtin_convertvector(x, vector8bool);
  // CHECK-LABEL: @int_tobool
  // CHECK-NOT: trunc <8 x i16> %{{[^ ]}} to <8 x i1>
  // CHECK: icmp ne <8 x i16> %{{[^ ]}}, zeroinitializer
}

vector8float int_tofp(vector8short x) {
  return __builtin_convertvector(x, vector8float);
  // CHECK-LABEL: @int_tofp
  // CHECK: sitofp <8 x i16> %{{[^ ]}} to <8 x float>
}

vector8float uint_tofp(vector8ushort x) {
  return __builtin_convertvector(x, vector8float);
  // CHECK-LABEL: @uint_tofp
  // CHECK: uitofp <8 x i16> %{{[^ ]}} to <8 x float>
}

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
template<typename T>
T int_toT(vector8long x) {
  return __builtin_convertvector(x, T);
}

extern "C" {
  vector8double int_toT_fp(vector8long x) {
    // CHECK-LABEL: @int_toT_fp
    // CHECK: sitofp <8 x i64> %{{[^ ]}} to <8 x double>
    return int_toT<vector8double>(x);
  }
}
#else
vector8double int_toT_fp(vector8long x) {
  return __builtin_convertvector(x, vector8double);
}
#endif

