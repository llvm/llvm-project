//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_CLC_H__
#define __CLC_OPENCL_CLC_H__

#ifndef cl_clang_storage_class_specifiers
#error Implementation requires cl_clang_storage_class_specifiers extension!
#endif

#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

/* Function Attributes */
#include <clc/clcfunc.h>

/* 6.1 Supported Data Types */
#include <clc/clctypes.h>

/* 6.2.3 Explicit Conversions */
#include <clc/opencl/convert.h>

/* 6.2.4.2 Reinterpreting Types Using as_type() and as_typen() */
#include <clc/opencl/as_type.h>

/* 6.11.1 Work-Item Functions */
#include <clc/opencl/workitem/get_global_id.h>
#include <clc/opencl/workitem/get_global_offset.h>
#include <clc/opencl/workitem/get_global_size.h>
#include <clc/opencl/workitem/get_group_id.h>
#include <clc/opencl/workitem/get_local_id.h>
#include <clc/opencl/workitem/get_local_size.h>
#include <clc/opencl/workitem/get_num_groups.h>
#include <clc/opencl/workitem/get_work_dim.h>

/* 6.11.2 Math Functions */
#include <clc/opencl/math/acos.h>
#include <clc/opencl/math/acosh.h>
#include <clc/opencl/math/acospi.h>
#include <clc/opencl/math/asin.h>
#include <clc/opencl/math/asinh.h>
#include <clc/opencl/math/asinpi.h>
#include <clc/opencl/math/atan.h>
#include <clc/opencl/math/atan2.h>
#include <clc/opencl/math/atan2pi.h>
#include <clc/opencl/math/atanh.h>
#include <clc/opencl/math/atanpi.h>
#include <clc/opencl/math/cbrt.h>
#include <clc/opencl/math/ceil.h>
#include <clc/opencl/math/copysign.h>
#include <clc/opencl/math/cos.h>
#include <clc/opencl/math/cosh.h>
#include <clc/opencl/math/cospi.h>
#include <clc/opencl/math/erf.h>
#include <clc/opencl/math/erfc.h>
#include <clc/opencl/math/exp.h>
#include <clc/opencl/math/exp10.h>
#include <clc/opencl/math/exp2.h>
#include <clc/opencl/math/expm1.h>
#include <clc/opencl/math/fabs.h>
#include <clc/opencl/math/fdim.h>
#include <clc/opencl/math/floor.h>
#include <clc/opencl/math/fma.h>
#include <clc/opencl/math/fmax.h>
#include <clc/opencl/math/fmin.h>
#include <clc/opencl/math/fmod.h>
#include <clc/opencl/math/fract.h>
#include <clc/opencl/math/frexp.h>
#include <clc/opencl/math/half_cos.h>
#include <clc/opencl/math/half_divide.h>
#include <clc/opencl/math/half_exp.h>
#include <clc/opencl/math/half_exp10.h>
#include <clc/opencl/math/half_exp2.h>
#include <clc/opencl/math/half_log.h>
#include <clc/opencl/math/half_log10.h>
#include <clc/opencl/math/half_log2.h>
#include <clc/opencl/math/half_powr.h>
#include <clc/opencl/math/half_recip.h>
#include <clc/opencl/math/half_rsqrt.h>
#include <clc/opencl/math/half_sin.h>
#include <clc/opencl/math/half_sqrt.h>
#include <clc/opencl/math/half_tan.h>
#include <clc/opencl/math/hypot.h>
#include <clc/opencl/math/ilogb.h>
#include <clc/opencl/math/ldexp.h>
#include <clc/opencl/math/lgamma.h>
#include <clc/opencl/math/lgamma_r.h>
#include <clc/opencl/math/log.h>
#include <clc/opencl/math/log10.h>
#include <clc/opencl/math/log1p.h>
#include <clc/opencl/math/log2.h>
#include <clc/opencl/math/logb.h>
#include <clc/opencl/math/mad.h>
#include <clc/opencl/math/maxmag.h>
#include <clc/opencl/math/minmag.h>
#include <clc/opencl/math/modf.h>
#include <clc/opencl/math/nan.h>
#include <clc/opencl/math/native_cos.h>
#include <clc/opencl/math/native_divide.h>
#include <clc/opencl/math/native_exp.h>
#include <clc/opencl/math/native_exp10.h>
#include <clc/opencl/math/native_exp2.h>
#include <clc/opencl/math/native_log.h>
#include <clc/opencl/math/native_log10.h>
#include <clc/opencl/math/native_log2.h>
#include <clc/opencl/math/native_powr.h>
#include <clc/opencl/math/native_recip.h>
#include <clc/opencl/math/native_rsqrt.h>
#include <clc/opencl/math/native_sin.h>
#include <clc/opencl/math/native_sqrt.h>
#include <clc/opencl/math/native_tan.h>
#include <clc/opencl/math/nextafter.h>
#include <clc/opencl/math/pow.h>
#include <clc/opencl/math/pown.h>
#include <clc/opencl/math/powr.h>
#include <clc/opencl/math/remainder.h>
#include <clc/opencl/math/remquo.h>
#include <clc/opencl/math/rint.h>
#include <clc/opencl/math/rootn.h>
#include <clc/opencl/math/round.h>
#include <clc/opencl/math/rsqrt.h>
#include <clc/opencl/math/sin.h>
#include <clc/opencl/math/sincos.h>
#include <clc/opencl/math/sinh.h>
#include <clc/opencl/math/sinpi.h>
#include <clc/opencl/math/sqrt.h>
#include <clc/opencl/math/tan.h>
#include <clc/opencl/math/tanh.h>
#include <clc/opencl/math/tanpi.h>
#include <clc/opencl/math/tgamma.h>
#include <clc/opencl/math/trunc.h>

/* 6.11.2.1 Floating-point macros */
#include <clc/float/definitions.h>

/* 6.11.3 Integer Functions */
#include <clc/opencl/integer/abs.h>
#include <clc/opencl/integer/abs_diff.h>
#include <clc/opencl/integer/add_sat.h>
#include <clc/opencl/integer/clz.h>
#include <clc/opencl/integer/ctz.h>
#include <clc/opencl/integer/hadd.h>
#include <clc/opencl/integer/mad24.h>
#include <clc/opencl/integer/mad_hi.h>
#include <clc/opencl/integer/mad_sat.h>
#include <clc/opencl/integer/mul24.h>
#include <clc/opencl/integer/mul_hi.h>
#include <clc/opencl/integer/popcount.h>
#include <clc/opencl/integer/rhadd.h>
#include <clc/opencl/integer/rotate.h>
#include <clc/opencl/integer/sub_sat.h>
#include <clc/opencl/integer/upsample.h>

/* 6.11.3 Integer Definitions */
#include <clc/integer/definitions.h>

/* 6.11.2 and 6.11.3 Shared Integer/Math Functions */
#include <clc/opencl/shared/clamp.h>
#include <clc/opencl/shared/max.h>
#include <clc/opencl/shared/min.h>
#include <clc/opencl/shared/vload.h>
#include <clc/opencl/shared/vstore.h>

/* 6.11.4 Common Functions */
#include <clc/opencl/common/degrees.h>
#include <clc/opencl/common/mix.h>
#include <clc/opencl/common/radians.h>
#include <clc/opencl/common/sign.h>
#include <clc/opencl/common/smoothstep.h>
#include <clc/opencl/common/step.h>

/* 6.11.5 Geometric Functions */
#include <clc/opencl/geometric/cross.h>
#include <clc/opencl/geometric/distance.h>
#include <clc/opencl/geometric/dot.h>
#include <clc/opencl/geometric/fast_distance.h>
#include <clc/opencl/geometric/fast_length.h>
#include <clc/opencl/geometric/fast_normalize.h>
#include <clc/opencl/geometric/length.h>
#include <clc/opencl/geometric/normalize.h>

/* 6.11.6 Relational Functions */
#include <clc/opencl/relational/all.h>
#include <clc/opencl/relational/any.h>
#include <clc/opencl/relational/bitselect.h>
#include <clc/opencl/relational/isequal.h>
#include <clc/opencl/relational/isfinite.h>
#include <clc/opencl/relational/isgreater.h>
#include <clc/opencl/relational/isgreaterequal.h>
#include <clc/opencl/relational/isinf.h>
#include <clc/opencl/relational/isless.h>
#include <clc/opencl/relational/islessequal.h>
#include <clc/opencl/relational/islessgreater.h>
#include <clc/opencl/relational/isnan.h>
#include <clc/opencl/relational/isnormal.h>
#include <clc/opencl/relational/isnotequal.h>
#include <clc/opencl/relational/isordered.h>
#include <clc/opencl/relational/isunordered.h>
#include <clc/opencl/relational/select.h>
#include <clc/opencl/relational/signbit.h>

#include <clc/opencl/synchronization/cl_mem_fence_flags.h>

/* 6.11.8 Synchronization Functions */
#include <clc/opencl/synchronization/barrier.h>

/* 6.11.9 Explicit Memory Fence Functions */
#include <clc/opencl/explicit_fence/explicit_memory_fence.h>

/* 6.11.10 Async Copy and Prefetch Functions */
#include <clc/opencl/async/async_work_group_copy.h>
#include <clc/opencl/async/async_work_group_strided_copy.h>
#include <clc/opencl/async/prefetch.h>
#include <clc/opencl/async/wait_group_events.h>

/* 6.11.11 Atomic Functions */
#include <clc/opencl/atomic/atomic_add.h>
#include <clc/opencl/atomic/atomic_and.h>
#include <clc/opencl/atomic/atomic_cmpxchg.h>
#include <clc/opencl/atomic/atomic_dec.h>
#include <clc/opencl/atomic/atomic_inc.h>
#include <clc/opencl/atomic/atomic_max.h>
#include <clc/opencl/atomic/atomic_min.h>
#include <clc/opencl/atomic/atomic_or.h>
#include <clc/opencl/atomic/atomic_sub.h>
#include <clc/opencl/atomic/atomic_xchg.h>
#include <clc/opencl/atomic/atomic_xor.h>

/* cl_khr_global_int32_base_atomics, cl_khr_local_int32_base_atomics and
 * cl_khr_int64_base_atomics Extension Functions */
#include <clc/opencl/atomic/atom_add.h>
#include <clc/opencl/atomic/atom_cmpxchg.h>
#include <clc/opencl/atomic/atom_dec.h>
#include <clc/opencl/atomic/atom_inc.h>
#include <clc/opencl/atomic/atom_sub.h>
#include <clc/opencl/atomic/atom_xchg.h>

/* cl_khr_global_int32_extended_atomics, cl_khr_local_int32_extended_atomics and
 * cl_khr_int64_extended_atomics Extension Functions */
#include <clc/opencl/atomic/atom_and.h>
#include <clc/opencl/atomic/atom_max.h>
#include <clc/opencl/atomic/atom_min.h>
#include <clc/opencl/atomic/atom_or.h>
#include <clc/opencl/atomic/atom_xor.h>

/* 6.12.12 Miscellaneous Vector Functions */
#include <clc/opencl/misc/shuffle.h>
#include <clc/opencl/misc/shuffle2.h>

/* 6.11.13 Image Read and Write Functions */
#include <clc/opencl/image/image.h>
#include <clc/opencl/image/image_defines.h>

#pragma OPENCL EXTENSION all : disable

#endif // __CLC_OPENCL_CLC_H__
