// RUN: %clangxx --cuda-device-only -S -emit-llvm -foffload-via-llvm \
// RUN:   --offload-arch=sm_90 -nocudalib --no-offloadlib \
// RUN:   -I %S/../../lib/Headers/llvm_offload_wrappers/gpu \
// RUN:   -I %S/../../lib/Headers/llvm_offload_wrappers/cuda \
// RUN:   -I %S/../../../offload/languages/include \
// RUN:   -I %S/../../../offload/languages/include/cuda \
// RUN:   -I %S/../../../offload/languages/include/hip \
// RUN:   -I %S/../../../offload/languages/kernel/include \
// RUN:   %s -o - | FileCheck --implicit-check-not=__nv_ --implicit-check-not=__clc_ %s

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


// The kernel above exercises the complete CUDA/HIP math declaration surface.
// Check representative calls from the libclc overload surface, including
// functions whose CUDA spelling uses an `f` suffix.

//CHECK: call {{.*}} @_Z{{[0-9]+}}acosd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}acoshd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}asind(
//CHECK: call {{.*}} @_Z{{[0-9]+}}asinhd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}atand(
//CHECK: call {{.*}} @_Z{{[0-9]+}}atan2dd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}atanhd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}cbrtd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}ceild(
//CHECK: call {{.*}} @_Z{{[0-9]+}}copysigndd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}cosd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}coshd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}erfd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}erfcd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}expd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}exp2d(
//CHECK: call {{.*}} @_Z{{[0-9]+}}exp10d(
//CHECK: call {{.*}} @_Z{{[0-9]+}}expm1d(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fabsd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fdimdd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}floord(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fmaddd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fmaxdd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fmindd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fmoddd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}frexpdPi(
//CHECK: call {{.*}} @_Z{{[0-9]+}}hypotdd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}ilogbd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}ldexpdi(
//CHECK: call {{.*}} @_Z{{[0-9]+}}lgammad(
//CHECK: call {{.*}} @_Z{{[0-9]+}}rintd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}roundd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}logd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}log10d(
//CHECK: call {{.*}} @_Z{{[0-9]+}}log1pd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}log2d(
//CHECK: call {{.*}} @_Z{{[0-9]+}}logbd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}modfdPd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}nextafterdd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}powdd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}remainderdd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}remquoddPi(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sind(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sincosdPd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sinpid(
//CHECK: call {{.*}} @_Z{{[0-9]+}}cospid(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sinhd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sqrtd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}tand(
//CHECK: call {{.*}} @_Z{{[0-9]+}}tanhd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}tgammad(
//CHECK: call {{.*}} @_Z{{[0-9]+}}truncd(
//CHECK: call {{.*}} @_Z{{[0-9]+}}acosf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}acoshf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}asinf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}asinhf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}atanf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}atan2ff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}atanhf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}cbrtf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}ceilf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}copysignff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}cosf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}coshf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}erff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}erfcf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}expf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}exp2f(
//CHECK: call {{.*}} @_Z{{[0-9]+}}exp10f(
//CHECK: call {{.*}} @_Z{{[0-9]+}}expm1f(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fabsf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fdimff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}floorf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fmafff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fmaxff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fminff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}fmodff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}frexpfPi(
//CHECK: call {{.*}} @_Z{{[0-9]+}}hypotff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}ilogbf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}ldexpfi(
//CHECK: call {{.*}} @_Z{{[0-9]+}}lgammaf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}rintf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}roundf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}logf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}log10f(
//CHECK: call {{.*}} @_Z{{[0-9]+}}log1pf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}log2f(
//CHECK: call {{.*}} @_Z{{[0-9]+}}logbf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}modffPf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}nextafterff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}powff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}remainderff(
//CHECK: call {{.*}} @_Z{{[0-9]+}}remquoffPi(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sinf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sincosfPf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sinpif(
//CHECK: call {{.*}} @_Z{{[0-9]+}}cospif(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sinhf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}sqrtf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}tanf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}tanhf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}tgammaf(
//CHECK: call {{.*}} @_Z{{[0-9]+}}truncf(
