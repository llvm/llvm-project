// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -fenable-ripple -fdisable-ripple-lib -emit-llvm %s -o - 2>&1 | FileCheck %s

#include <ripple.h>
#include <ripple_math.h>
// These are simply defined to play well with "gen_check_unary_mathfn"
#define isnanf isnan
#define isinff isinf

void check_mylog(const float x[128], float y[128]) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  int v0 = ripple_id(BS, 0);
  y[v0] = logf(x[v0]);
  // CHECK: @check_mylog
  // CHECK: <128 x float> @llvm.log.v128f32(<128 x float> %{{[0-9]+}})
  // CHECK: ret
}

#define gen_check_unary_mathfn(N, OP, LONGTYPE)                                \
  void check_##OP(const LONGTYPE x[N], LONGTYPE y[N]) {                        \
    ripple_block_t BS = ripple_set_block_shape(0, N);                                               \
    int v0 = ripple_id(BS, 0);                                                  \
    y[v0] = OP(x[v0]);                                                         \
  }

gen_check_unary_mathfn(32, sqrtf, float);
// CHECK: @check_sqrtf
// CHECK: <32 x float> @llvm.sqrt.v32f32(<32 x float> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(16, sqrt, double);
// CHECK: @check_sqrt
// CHECK: <16 x double> @llvm.sqrt.v16f64(<16 x double> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(64, sqrtf16, _Float16);
// CHECK: @check_sqrtf16
// CHECK: <64 x half> @llvm.sqrt.v64f16(<64 x half> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(32, sinf, float);
// CHECK: @check_sinf
// CHECK: <32 x float> @llvm.sin.v32f32(<32 x float> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(16, sin, double);
// CHECK: @check_sin
// CHECK: <16 x double> @llvm.sin.v16f64(<16 x double> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(64, sinf16, _Float16);
// CHECK: @check_sinf16
// CHECK: <64 x half> @llvm.sin.v64f16(<64 x half> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(32, cosf, float);
// CHECK: @check_cosf
// CHECK: <32 x float> @llvm.cos.v32f32(<32 x float> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(16, cos, double);
// CHECK: @check_cos
// CHECK: <16 x double> @llvm.cos.v16f64(<16 x double> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(64, cosf16, _Float16);
// CHECK: @check_cosf16
// CHECK: <64 x half> @llvm.cos.v64f16(<64 x half> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(32, expf, float);
// CHECK: @check_expf
// CHECK: <32 x float> @llvm.exp.v32f32(<32 x float> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(16, exp, double);
// CHECK: @check_exp
// CHECK: <16 x double> @llvm.exp.v16f64(<16 x double> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(64, expf16, _Float16);
// CHECK: @check_expf16
// CHECK: <64 x half> @llvm.exp.v64f16(<64 x half> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(32, exp2f, float);
// CHECK: @check_exp2f
// CHECK: <32 x float> @llvm.exp2.v32f32(<32 x float> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(16, exp2, double);
// CHECK: @check_exp2
// CHECK: <16 x double> @llvm.exp2.v16f64(<16 x double> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(64, exp2f16, _Float16);
// CHECK: @check_exp2f16
// CHECK: <64 x half> @llvm.exp2.v64f16(<64 x half> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(32, exp10f, float);
// CHECK: @check_exp10f
// CHECK: <32 x float> @llvm.exp10.v32f32(<32 x float> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(16, exp10, double);
// CHECK: @check_exp10
// CHECK: <16 x double> @llvm.exp10.v16f64(<16 x double> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(64, exp10f16, _Float16);
// CHECK: @check_exp10f16
// CHECK: <64 x half> @llvm.exp10.v64f16(<64 x half> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(32, logf, float);
// CHECK: @check_logf
// CHECK: <32 x float> @llvm.log.v32f32(<32 x float> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(16, log, double);
// CHECK: @check_log
// CHECK: <16 x double> @llvm.log.v16f64(<16 x double> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(64, logf16, _Float16);
// CHECK: @check_logf16
// CHECK: <64 x half> @llvm.log.v64f16(<64 x half> %{{[0-9]+}})
// CHECK: ret

gen_check_unary_mathfn(32, isnanf, float);
// CHECK: @check_isnanf
// CHECK: <32 x i1> @llvm.is.fpclass.v32f32(<32 x float> %{{[0-9]+}}, i32 3)
// CHECK: ret

gen_check_unary_mathfn(16, isnan, double);
// CHECK: @check_isnan
// CHECK: <16 x i1> @llvm.is.fpclass.v16f64(<16 x double> %{{[0-9]+}}, i32 3)
// CHECK: ret

gen_check_unary_mathfn(32, isinff, float);
// CHECK: @check_isinff
// CHECK: <32 x i1> @llvm.is.fpclass.v32f32(<32 x float> %{{[0-9]+}}, i32 516)
// CHECK: ret

gen_check_unary_mathfn(16, isinf, double);
// CHECK: @check_isinf
// CHECK: <16 x i1> @llvm.is.fpclass.v16f64(<16 x double> %{{[0-9]+}}, i32 516)
// CHECK: ret
