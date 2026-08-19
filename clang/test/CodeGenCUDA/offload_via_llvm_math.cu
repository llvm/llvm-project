// RUN: %clangxx --cuda-device-only -S -emit-llvm -foffload-via-llvm \
// RUN:   --offload-arch=sm_90 -nocudalib \
// RUN:   -I %S/../../../offload/languages/include \
// RUN:   -I %S/../../../offload/languages/include/cuda \
// RUN:   -I %S/../../../offload/languages/include/hip \
// RUN:   -I %S/../../../offload/languages/kernel/include \
// RUN:   %s -o - | FileCheck %s

#include "cuda_runtime.h"

__global__ void math_kernel(double *D, float *F, int *I, long *L,
                            long long *LL) {
  D[0] = acos(D[0]);
  D[1] = acosh(D[1]);
  D[2] = asin(D[2]);
  D[3] = asinh(D[3]);
  D[4] = atan(D[4]);
  D[5] = atan2(D[5], D[6]);
  D[6] = atanh(D[6]);
  D[7] = cbrt(D[7]);
  D[8] = ceil(D[8]);
  D[9] = copysign(D[9], D[10]);
  D[10] = cos(D[10]);
  D[11] = cosh(D[11]);
  D[12] = erf(D[12]);
  D[13] = erfc(D[13]);
  D[14] = exp(D[14]);
  D[15] = exp2(D[15]);
  D[16] = exp10(D[16]);
  D[17] = expm1(D[17]);
  D[18] = fabs(D[18]);
  D[19] = fdim(D[19], D[20]);
  D[20] = floor(D[20]);
  D[21] = fma(D[21], D[22], D[23]);
  D[22] = fmax(D[22], D[23]);
  D[23] = fmin(D[23], D[24]);
  D[24] = fmod(D[24], D[25]);
  D[25] = frexp(D[25], I);
  D[26] = hypot(D[26], D[27]);
  I[1] = ilogb(D[27]);
  D[28] = ldexp(D[28], I[1]);
  D[29] = lgamma(D[29]);
  LL[0] = llrint(D[30]);
  LL[1] = llround(D[31]);
  D[30] = log(D[30]);
  D[31] = log10(D[31]);
  D[32] = log1p(D[32]);
  D[33] = log2(D[33]);
  D[34] = logb(D[34]);
  L[0] = lrint(D[35]);
  L[1] = lround(D[36]);
  D[35] = modf(D[35], &D[36]);
  D[36] = nearbyint(D[36]);
  D[37] = nextafter(D[37], D[38]);
  D[38] = pow(D[38], D[39]);
  D[39] = remainder(D[39], D[40]);
  D[40] = remquo(D[40], D[41], &I[2]);
  D[41] = rint(D[41]);
  D[42] = round(D[42]);
  D[43] = roundeven(D[43]);
  D[44] = scalbln(D[44], L[2]);
  D[45] = scalbn(D[45], I[3]);
  D[46] = sin(D[46]);
  sincos(D[47], &D[47], &D[48]);
  sincospi(D[49], &D[49], &D[50]);
  D[51] = sinh(D[51]);
  D[52] = sqrt(D[52]);
  D[53] = tan(D[53]);
  D[54] = tanh(D[54]);
  D[55] = tgamma(D[55]);
  D[56] = trunc(D[56]);

  F[0] = acosf(F[0]);
  F[1] = acoshf(F[1]);
  F[2] = asinf(F[2]);
  F[3] = asinhf(F[3]);
  F[4] = atanf(F[4]);
  F[5] = atan2f(F[5], F[6]);
  F[6] = atanhf(F[6]);
  F[7] = cbrtf(F[7]);
  F[8] = ceilf(F[8]);
  F[9] = copysignf(F[9], F[10]);
  F[10] = cosf(F[10]);
  F[11] = coshf(F[11]);
  F[12] = erff(F[12]);
  F[13] = erfcf(F[13]);
  F[14] = expf(F[14]);
  F[15] = exp2f(F[15]);
  F[16] = exp10f(F[16]);
  F[17] = expm1f(F[17]);
  F[18] = fabsf(F[18]);
  F[19] = fdimf(F[19], F[20]);
  F[20] = floorf(F[20]);
  F[21] = fmaf(F[21], F[22], F[23]);
  F[22] = fmaxf(F[22], F[23]);
  F[23] = fminf(F[23], F[24]);
  F[24] = fmodf(F[24], F[25]);
  F[25] = frexpf(F[25], &I[4]);
  F[26] = hypotf(F[26], F[27]);
  I[5] = ilogbf(F[27]);
  F[28] = ldexpf(F[28], I[5]);
  F[29] = lgammaf(F[29]);
  LL[2] = llrintf(F[30]);
  LL[3] = llroundf(F[31]);
  F[30] = logf(F[30]);
  F[31] = log10f(F[31]);
  F[32] = log1pf(F[32]);
  F[33] = log2f(F[33]);
  F[34] = logbf(F[34]);
  L[3] = lrintf(F[35]);
  L[4] = lroundf(F[36]);
  F[35] = modff(F[35], &F[36]);
  F[36] = nearbyintf(F[36]);
  F[37] = nextafterf(F[37], F[38]);
  F[38] = powf(F[38], F[39]);
  F[39] = remainderf(F[39], F[40]);
  F[40] = remquof(F[40], F[41], &I[6]);
  F[41] = rintf(F[41]);
  F[42] = roundf(F[42]);
  F[43] = roundevenf(F[43]);
  F[44] = scalblnf(F[44], L[5]);
  F[45] = scalbnf(F[45], I[7]);
  F[46] = sinf(F[46]);
  sincosf(F[47], &F[47], &F[48]);
  sincospif(F[49], &F[49], &F[50]);
  F[51] = sinhf(F[51]);
  F[52] = sqrtf(F[52]);
  F[53] = tanf(F[53]);
  F[54] = tanhf(F[54]);
  F[55] = tgammaf(F[55]);
  F[56] = truncf(F[56]);
}

// CHECK-NOT: __nv_
// CHECK: call {{.*}} @acos(
// CHECK: call {{.*}} @acosh(
// CHECK: call {{.*}} @asin(
// CHECK: call {{.*}} @asinh(
// CHECK: call {{.*}} @atan(
// CHECK: call {{.*}} @atan2(
// CHECK: call {{.*}} @atanh(
// CHECK: call {{.*}} @cbrt(
// CHECK: call {{.*}} @ceil(
// CHECK: call {{.*}} @copysign(
// CHECK: call {{.*}} @cos(
// CHECK: call {{.*}} @cosh(
// CHECK: call {{.*}} @erf(
// CHECK: call {{.*}} @erfc(
// CHECK: call {{.*}} @exp(
// CHECK: call {{.*}} @exp2(
// CHECK: call {{.*}} @exp10(
// CHECK: call {{.*}} @expm1(
// CHECK: call {{.*}} @fabs(
// CHECK: call {{.*}} @fdim(
// CHECK: call {{.*}} @floor(
// CHECK: call {{.*}} @fma(
// CHECK: call {{.*}} @fmax(
// CHECK: call {{.*}} @fmin(
// CHECK: call {{.*}} @fmod(
// CHECK: call {{.*}} @frexp(
// CHECK: call {{.*}} @hypot(
// CHECK: call {{.*}} @ilogb(
// CHECK: call {{.*}} @ldexp(
// CHECK: call {{.*}} @lgamma(
// CHECK: call {{.*}} @llrint(
// CHECK: call {{.*}} @llround(
// CHECK: call {{.*}} @log(
// CHECK: call {{.*}} @log10(
// CHECK: call {{.*}} @log1p(
// CHECK: call {{.*}} @log2(
// CHECK: call {{.*}} @logb(
// CHECK: call {{.*}} @lrint(
// CHECK: call {{.*}} @lround(
// CHECK: call {{.*}} @modf(
// CHECK: call {{.*}} @nearbyint(
// CHECK: call {{.*}} @nextafter(
// CHECK: call {{.*}} @pow(
// CHECK: call {{.*}} @remainder(
// CHECK: call {{.*}} @remquo(
// CHECK: call {{.*}} @rint(
// CHECK: call {{.*}} @round(
// CHECK: call {{.*}} @roundeven(
// CHECK: call {{.*}} @scalbln(
// CHECK: call {{.*}} @scalbn(
// CHECK: call {{.*}} @sin(
// CHECK: call {{.*}} @sincos(
// CHECK: call {{.*}} @sincospi(
// CHECK: call {{.*}} @sinh(
// CHECK: call {{.*}} @sqrt(
// CHECK: call {{.*}} @tan(
// CHECK: call {{.*}} @tanh(
// CHECK: call {{.*}} @tgamma(
// CHECK: call {{.*}} @trunc(
// CHECK: call {{.*}} @acosf(
// CHECK: call {{.*}} @acoshf(
// CHECK: call {{.*}} @asinf(
// CHECK: call {{.*}} @asinhf(
// CHECK: call {{.*}} @atanf(
// CHECK: call {{.*}} @atan2f(
// CHECK: call {{.*}} @atanhf(
// CHECK: call {{.*}} @cbrtf(
// CHECK: call {{.*}} @ceilf(
// CHECK: call {{.*}} @copysignf(
// CHECK: call {{.*}} @cosf(
// CHECK: call {{.*}} @coshf(
// CHECK: call {{.*}} @erff(
// CHECK: call {{.*}} @erfcf(
// CHECK: call {{.*}} @expf(
// CHECK: call {{.*}} @exp2f(
// CHECK: call {{.*}} @exp10f(
// CHECK: call {{.*}} @expm1f(
// CHECK: call {{.*}} @fabsf(
// CHECK: call {{.*}} @fdimf(
// CHECK: call {{.*}} @floorf(
// CHECK: call {{.*}} @fmaf(
// CHECK: call {{.*}} @fmaxf(
// CHECK: call {{.*}} @fminf(
// CHECK: call {{.*}} @fmodf(
// CHECK: call {{.*}} @frexpf(
// CHECK: call {{.*}} @hypotf(
// CHECK: call {{.*}} @ilogbf(
// CHECK: call {{.*}} @ldexpf(
// CHECK: call {{.*}} @lgammaf(
// CHECK: call {{.*}} @llrintf(
// CHECK: call {{.*}} @llroundf(
// CHECK: call {{.*}} @logf(
// CHECK: call {{.*}} @log10f(
// CHECK: call {{.*}} @log1pf(
// CHECK: call {{.*}} @log2f(
// CHECK: call {{.*}} @logbf(
// CHECK: call {{.*}} @lrintf(
// CHECK: call {{.*}} @lroundf(
// CHECK: call {{.*}} @modff(
// CHECK: call {{.*}} @nearbyintf(
// CHECK: call {{.*}} @nextafterf(
// CHECK: call {{.*}} @powf(
// CHECK: call {{.*}} @remainderf(
// CHECK: call {{.*}} @remquof(
// CHECK: call {{.*}} @rintf(
// CHECK: call {{.*}} @roundf(
// CHECK: call {{.*}} @roundevenf(
// CHECK: call {{.*}} @scalblnf(
// CHECK: call {{.*}} @scalbnf(
// CHECK: call {{.*}} @sinf(
// CHECK: call {{.*}} @sincosf(
// CHECK: call {{.*}} @sincospif(
// CHECK: call {{.*}} @sinhf(
// CHECK: call {{.*}} @sqrtf(
// CHECK: call {{.*}} @tanf(
// CHECK: call {{.*}} @tanhf(
// CHECK: call {{.*}} @tgammaf(
// CHECK: call {{.*}} @truncf(
