/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


/*
 * Define supported architecutres.
 */
typedef enum {
	/*
	 * arch_any is only intended to be use in defining the dispatch
	 * table definitions.  Its purpose is to simplify having to define
	 * the same set of jump table entries that are common to all processor
	 * platforms.
	 */
	arch_any=0,
#if defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
	arch_em64t,	// em64t/opteron
	arch_sse4,	// SSE4A/SSE4.1
			// greyhound, barcelona, core2,
			// istanbul, nehalem, penryn, shanghai
	arch_avx,	// AVX 128 Intel sandybridge
	arch_avxfma4,	// AVX 128 AMD bulldozer, piledriver
	arch_avx2, 	// AVX2 256 haswell
	arch_avx512knl, // AVX512, knights landing
	arch_avx512,	// AVX512, skylake
#elif defined(TARGET_LINUX_POWER)
	arch_p8,	// Power8
	arch_p9,	// Power9
#elif defined(TARGET_ARM64)
	arch_armv8,     // ARM V8
	arch_armv81a,   // ARM V8.1-A
	arch_armv82,    // ARM V8.2
#else
	arch_generic,  // Generic CPU
#endif
	arch_size=16,	// *** Always last
} arch_e;

/*
 * Define scalar and vector formats.
 */

typedef enum {
	sv_ss=0,	// single scalar
	sv_ds,		// double scalar
	sv_qs, 		// quad scalar
	sv_cs,		// single complex - C ABI
	sv_zs,		// double complex - C ABI
	sv_cv1,		// single complex - vector

	// 128-bit
	sv_sv4,		// single vector
	sv_dv2,		// double vector
	sv_cv2,		// single complex vector
	sv_zv1,		// double complex vector

	// 256-bit
	sv_sv8,		// single vector
	sv_dv4,		// double vector
	sv_cv4,		// single complex vector
	sv_zv2,		// double complex vector

	// 512-bit
	sv_sv16,	// single vector
	sv_dv8,		// double vector
	sv_cv8,		// single complex vector
	sv_zv4,		// double complex vector

	// 128-bit - Masked
	sv_sv4m,	// single vector
	sv_dv2m,	// double vector
	sv_cv2m,	// single complex vector
	sv_zv1m,	// double complex vector

	// 256-bit - Masked
	sv_sv8m,	// single vector
	sv_dv4m,	// double vector
	sv_cv4m,	// single complex vector
	sv_zv2m,	// double complex vector

	// 512-bit - Masked
	sv_sv16m,	// single vector
	sv_dv8m,	// double vector
	sv_cv8m,	// single complex vector
	sv_zv4m,	// double complex vector

	sv_size=64,	// *** Always last
} sv_e;


/*
 * Define fast/relaxed/precise classes.
 */
typedef enum {
	frp_f=0,	// fast
	frp_r,		// relaxed
	frp_p,		// precise
	frp_s,		// Sleef
	frp_size,	// *** Always last
} frp_e;

/*
 * Define intrinsic funtions.
 */
typedef enum {
	func_acos=0,
	func_asin,
	func_atan,
	func_atan2,
	func_cos,
	func_sin,
	func_tan,
	func_cosh,
	func_sinh,
	func_tanh,
	func_exp,
	func_log,
	func_log10,
	func_pow,
	func_powi1,	// R{4,8}*I4
	func_powi,	// R{4,8}*I4(:)
	func_powk1,	// R{4,8}*I8
	func_powk,	// R{4,8}*I8(:)
	func_sincos,	// Returns pair of values
	func_div,	// division
	func_sqrt,	// square root
	func_mod,	// mod(R{4,8},R{4,8})
	func_aint,
	func_ceil,
	func_floor,
	func_size,
} func_e;

/*
 *	Elements sizes;
 */
typedef	enum	{
	elmtsz_32 = 0,	// ss,   cs
	elmtsz_64,	// ds,   zs,  cv1
	elmtsz_128,	// sv4,  dv2, cv2, zv1
	elmtsz_256,	// sv8,  dv4, cv4, zv2
	elmtsz_512,	// sv16, dv8, cv8, zv4
	elmtsz_size
} elmtsz_e;

typedef	void(*p2f)();

typedef	struct	{
	arch_e	arch;	// Architecture
	func_e	func;	// Function
	sv_e	sv;	// Scalar/vector type
	p2f	pf;	// Pointer to fast
	p2f	pr;	// Pointer to relaxed
	p2f	pp;	// Pointer to precise
	p2f	ps;	// Pointer to Sleef
} mth_intrins_defs_t;

extern	p2f	__mth_rt_vi_ptrs[func_size][sv_size][frp_size];
extern	p2f	__mth_rt_vi_ptrs_stat[func_size][sv_size][frp_size];
//extern	p2f	__mth_rt_intrins_ptrs[func_size][sv_size][frp_size];

// _func: function
// _sv: scalar/vector types
// _a: architecture
// _f: name for fast
// _r: name for relaxed
// _p: name for precise
// _s: name for Sleef
#define	MTHINTRIN(_func, _sv, _a, _f, _r, _p, _s) \
extern void _f (void); \
extern void _r (void); \
extern void _p (void); \
extern void _s (void);

#define MTH_DISPATCH_FUNC(f)    f
#define	MTH_DISPATCH_TBL	__mth_rt_vi_ptrs

#define	_MTH_I_INIT()
#ifdef	MTH_I_INTRIN_INIT
#undef	MTH_DISPATCH_FUNC
#define	MTH_DISPATCH_FUNC(f)	f##_init
#undef	_MTH_I_INIT
#define	_MTH_I_INIT()	(void) __math_dispatch_init()
extern	void __math_dispatch_init(void);
#endif

#ifdef	MTH_I_INTRIN_STATS
#undef	MTH_DISPATCH_FUNC
#define	MTH_DISPATCH_FUNC(f)	f##_prof
#undef	MTH_DISPATCH_TBL
#define	MTH_DISPATCH_TBL	__mth_rt_vi_ptrs_stat
/*
 * Unsigned integers of number of calls.
 *
 * XXX - Layout of the table is *different* than __mth_rt_vi_ptrs.
 * TBD which layout is better.
 */
extern	uint64_t	__mth_rt_stats[frp_size][func_size][sv_size];
#define	_MTH_I_STATS_INC(_func,_sv,_frp) \
	(void)__sync_fetch_and_add(&__mth_rt_stats[_frp][_func][_sv], 1);
#else	// MTH_I_INTRIN_STATS
#define	_MTH_I_STATS_INC(_func,_sv,_frp)
#endif	// MTH_I_INTRIN_STATS
