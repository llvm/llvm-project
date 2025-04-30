
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


/*
 * Real.
 */

typedef double  vrd1_t;
typedef double  vrd2_t  __attribute__((vector_size(2*sizeof(double))));
typedef double  vrd4_t  __attribute__((vector_size(4*sizeof(double))));
typedef double  vrd8_t  __attribute__((vector_size(8*sizeof(double))));
typedef	float	vrs1_t;
typedef	float	vrs4_t	__attribute__((vector_size(4*sizeof(float))));
typedef	float	vrs8_t	__attribute__((vector_size(8*sizeof(float))));
typedef	float	vrs16_t	__attribute__((vector_size(16*sizeof(float))));


/*
 * Complex.
 *
 * Note:
 * Vector structures cannot be made up of structures contaning real and
 * imaginary components.
 * As such, complex vector structures are in name only and simply
 * overloaded to the REALs.  To extract the R and i's, other macros or
 * C constructs must be used.
 */

typedef double  vcd1_t  __attribute__((vector_size(2*sizeof(double))));
typedef double  vcd2_t  __attribute__((vector_size(4*sizeof(double))));
typedef double  vcd4_t  __attribute__((vector_size(8*sizeof(double))));
typedef float   vcs1_t  __attribute__((vector_size(2*sizeof(float))));
typedef float   vcs2_t  __attribute__((vector_size(4*sizeof(float))));
typedef float   vcs4_t  __attribute__((vector_size(8*sizeof(float))));
typedef float   vcs8_t  __attribute__((vector_size(16*sizeof(float))));


/*
 * Integer.
 */

typedef	int32_t	vis1_t;
typedef	int32_t	vis2_t	__attribute__((vector_size(2*sizeof(int32_t))));
typedef	int32_t	vis4_t	__attribute__((vector_size(4*sizeof(int32_t))));
typedef	int32_t	vis8_t	__attribute__((vector_size(8*sizeof(int32_t))));
typedef	int32_t	vis16_t	__attribute__((vector_size(16*sizeof(int32_t))));
typedef	int64_t	vid1_t;
typedef	int64_t	vid2_t	__attribute__((vector_size(2*sizeof(int64_t))));
typedef	int64_t	vid4_t	__attribute__((vector_size(4*sizeof(int64_t))));
typedef	int64_t	vid8_t	__attribute__((vector_size(8*sizeof(int64_t))));

#define _CONCAT2(a,b)    a##b
#define CONCAT2(a,b) _CONCAT(a,b)
#define _CONCAT3(a,b,c)    a##b##c
#define CONCAT3(a,b,c) _CONCAT3(a,b,c)
#define _CONCAT4(a,b,c,d)    a##b##c##d
#define CONCAT4(a,b,c,d) _CONCAT4(a,b,c,d)
#define _CONCAT5(a,b,c,d,e)    a##b##c##d##e
#define CONCAT5(a,b,c,d,e) _CONCAT5(a,b,c,d,e)
#define _CONCAT6(a,b,c,d,e,f)    a##b##c##d##e##f
#define CONCAT6(a,b,c,d,e,f) _CONCAT6(a,b,c,d,e,f)
#define _CONCAT7(a,b,c,d,e,f,g)    a##b##c##d##e##f##g
#define CONCAT7(a,b,c,d,e,f,g) _CONCAT7(a,b,c,d,e,f,g)
#define _CONCAT8(a,b,c,d,e,f,g,h)    a##b##c##d##e##f##g##h
#define CONCAT8(a,b,c,d,e,f,g,h) _CONCAT8(a,b,c,d,e,f,g,h)

#define _STRINGIFY(_n) #_n
#define STRINGIFY(_n) _STRINGIFY(_n)


#if ! defined(MAX_VREG_SIZE)
#error  MAX_VREG_SIZE must be defined.
#endif


#if MAX_VREG_SIZE == 64
#define VLS     1
#define VLD     1
#define VIS_T   vis1_t
#define VID_T   vid1_t
#define VRS_T   vrs1_t
#define VRD_T   vrd1_t
#define FMIN	1.0f
#define DMIN	1.0d
#define VRET(subscript) vret
#define ROUT(subscript) rout
#define ROUTM(subscript) routm
#define RES(subscript) res
#define EXP(subscript) exp
#define VVMASK(subscript) vvmask
#elif MAX_VREG_SIZE == 128
#define VLS     4
#define VLD     2
#define VIS_T   vis4_t
#define VID_T   vid2_t
#define VRS_T   vrs4_t
#define VRD_T   vrd2_t
#define FMIN	2.0f
#define DMIN	2.0d
#define VRET(subscript) vret[subscript]
#define ROUT(subscript) rout[subscript]
#define ROUTM(subscript) routm[subscript]
#define RES(subscript) res[subscript]
#define EXP(subscript) exp[subscript]
#define VVMASK(subscript) vvmask[subscript]
#elif   MAX_VREG_SIZE == 256
#define VLS     8
#define VLD     4
#define VIS_T   vis8_t
#define VID_T   vid4_t
#define VRS_T   vrs8_t
#define VRD_T   vrd4_t
#define FMIN	6.0f
#define DMIN	6.0d
#define VRET(subscript) vret[subscript]
#define ROUT(subscript) rout[subscript]
#define ROUTM(subscript) routm[subscript]
#define RES(subscript) res[subscript]
#define EXP(subscript) exp[subscript]
#define VVMASK(subscript) vvmask[subscript]
#elif   MAX_VREG_SIZE == 512
#define VLS     16
#define VLD     8
#define VIS_T   vis16_t
#define VID_T   vid8_t
#define VRS_T   vrs16_t
#define VRD_T   vrd8_t
#define FMIN	14.0f
#define DMIN	14.0d
#define VRET(subscript) vret[subscript]
#define ROUT(subscript) rout[subscript]
#define ROUTM(subscript) routm[subscript]
#define RES(subscript) res[subscript]
#define EXP(subscript) exp[subscript]
#define VVMASK(subscript) vvmask[subscript]
#else
#error  MAX_VREG_SIZE must be one of 64, 128, 256, or 512
#endif

#define FCONST1 0.0f
#define FCONST2 31.0f
#define DCONST1 0.0d
#define DCONST2 31.0d

#define EXTERN_EFUNC(name) \
    extern VRS_T \
    CONCAT5(__fs_,name,_,VLS,)(VRS_T), CONCAT5(__fs_,name,_,VLS,m)(VRS_T,VIS_T), \
    CONCAT5(__rs_,name,_,VLS,)(VRS_T), CONCAT5(__rs_,name,_,VLS,m)(VRS_T,VIS_T), \
    CONCAT5(__ps_,name,_,VLS,)(VRS_T), CONCAT5(__ps_,name,_,VLS,m)(VRS_T,VIS_T); \
    extern VRD_T \
    CONCAT5(__fd_,name,_,VLD,)(VRD_T), CONCAT5(__fd_,name,_,VLD,m)(VRD_T,VID_T), \
    CONCAT5(__rd_,name,_,VLD,)(VRD_T), CONCAT5(__rd_,name,_,VLD,m)(VRD_T,VID_T), \
    CONCAT5(__pd_,name,_,VLD,)(VRD_T), CONCAT5(__pd_,name,_,VLD,m)(VRD_T,VID_T)

#define EXTERN_EFUNC2(name) \
    extern VRS_T \
    CONCAT5(__fs_,name,_,VLS,)(VRS_T,VRS_T), CONCAT5(__fs_,name,_,VLS,m)(VRS_T,VRS_T,VIS_T), \
    CONCAT5(__rs_,name,_,VLS,)(VRS_T,VRS_T), CONCAT5(__rs_,name,_,VLS,m)(VRS_T,VRS_T,VIS_T), \
    CONCAT5(__ps_,name,_,VLS,)(VRS_T,VRS_T), CONCAT5(__ps_,name,_,VLS,m)(VRS_T,VRS_T,VIS_T); \
    extern VRD_T \
    CONCAT5(__fd_,name,_,VLD,)(VRD_T,VRD_T), CONCAT5(__fd_,name,_,VLD,m)(VRD_T,VRD_T,VID_T), \
    CONCAT5(__rd_,name,_,VLD,)(VRD_T,VRD_T), CONCAT5(__rd_,name,_,VLD,m)(VRD_T,VRD_T,VID_T), \
    CONCAT5(__pd_,name,_,VLD,)(VRD_T,VRD_T), CONCAT5(__pd_,name,_,VLD,m)(VRD_T,VRD_T,VID_T)


#define EXTERN_EFUNC2i(name) \
    extern VRS_T \
    CONCAT5(__fs_,name,_,VLS,)(VRS_T,VIS_T), CONCAT5(__fs_,name,_,VLS,m)(VRS_T,VIS_T,VIS_T), \
    CONCAT5(__rs_,name,_,VLS,)(VRS_T,VIS_T), CONCAT5(__rs_,name,_,VLS,m)(VRS_T,VIS_T,VIS_T), \
    CONCAT5(__ps_,name,_,VLS,)(VRS_T,VIS_T), CONCAT5(__ps_,name,_,VLS,m)(VRS_T,VIS_T,VIS_T); \
    extern VRD_T \
    CONCAT5(__fd_,name,_,VLD,)(VRD_T,VIS_T), CONCAT5(__fd_,name,_,VLD,m)(VRD_T,VIS_T,VID_T), \
    CONCAT5(__rd_,name,_,VLD,)(VRD_T,VIS_T), CONCAT5(__rd_,name,_,VLD,m)(VRD_T,VIS_T,VID_T), \
    CONCAT5(__pd_,name,_,VLD,)(VRD_T,VIS_T), CONCAT5(__pd_,name,_,VLD,m)(VRD_T,VIS_T,VID_T)


#define EXTERN_EFUNC2i1(name) \
    extern VRS_T \
    CONCAT5(__fs_,name,_,VLS,)(VRS_T,int32_t), CONCAT5(__fs_,name,_,VLS,m)(VRS_T,int32_t,VIS_T), \
    CONCAT5(__rs_,name,_,VLS,)(VRS_T,int32_t), CONCAT5(__rs_,name,_,VLS,m)(VRS_T,int32_t,VIS_T), \
    CONCAT5(__ps_,name,_,VLS,)(VRS_T,int32_t), CONCAT5(__ps_,name,_,VLS,m)(VRS_T,int32_t,VIS_T); \
    extern VRD_T \
    CONCAT5(__fd_,name,_,VLD,)(VRD_T,int32_t), CONCAT5(__fd_,name,_,VLD,m)(VRD_T,int32_t,VID_T), \
    CONCAT5(__rd_,name,_,VLD,)(VRD_T,int32_t), CONCAT5(__rd_,name,_,VLD,m)(VRD_T,int32_t,VID_T), \
    CONCAT5(__pd_,name,_,VLD,)(VRD_T,int32_t), CONCAT5(__pd_,name,_,VLD,m)(VRD_T,int32_t,VID_T)


#define EXTERN_EFUNC2k(name) \
    extern VRS_T \
    CONCAT5(__fs_,name,_,VLS,)(VRS_T,VID_T), CONCAT5(__fs_,name,_,VLS,m)(VRS_T,VID_T,VIS_T), \
    CONCAT5(__rs_,name,_,VLS,)(VRS_T,VID_T), CONCAT5(__rs_,name,_,VLS,m)(VRS_T,VID_T,VIS_T), \
    CONCAT5(__ps_,name,_,VLS,)(VRS_T,VID_T), CONCAT5(__ps_,name,_,VLS,m)(VRS_T,VID_T,VIS_T); \
    extern VRD_T \
    CONCAT5(__fd_,name,_,VLD,)(VRD_T,VID_T), CONCAT5(__fd_,name,_,VLD,m)(VRD_T,VID_T,VID_T), \
    CONCAT5(__rd_,name,_,VLD,)(VRD_T,VID_T), CONCAT5(__rd_,name,_,VLD,m)(VRD_T,VID_T,VID_T), \
    CONCAT5(__pd_,name,_,VLD,)(VRD_T,VID_T), CONCAT5(__pd_,name,_,VLD,m)(VRD_T,VID_T,VID_T)


#define EXTERN_EFUNC2k1(name) \
    extern VRS_T \
    CONCAT5(__fs_,name,_,VLS,)(VRS_T,int64_t), CONCAT5(__fs_,name,_,VLS,m)(VRS_T,int64_t,VIS_T), \
    CONCAT5(__rs_,name,_,VLS,)(VRS_T,int64_t), CONCAT5(__rs_,name,_,VLS,m)(VRS_T,int64_t,VIS_T), \
    CONCAT5(__ps_,name,_,VLS,)(VRS_T,int64_t), CONCAT5(__ps_,name,_,VLS,m)(VRS_T,int64_t,VIS_T); \
    extern VRD_T \
    CONCAT5(__fd_,name,_,VLD,)(VRD_T,int64_t), CONCAT5(__fd_,name,_,VLD,m)(VRD_T,int64_t,VID_T), \
    CONCAT5(__rd_,name,_,VLD,)(VRD_T,int64_t), CONCAT5(__rd_,name,_,VLD,m)(VRD_T,int64_t,VID_T), \
    CONCAT5(__pd_,name,_,VLD,)(VRD_T,int64_t), CONCAT5(__pd_,name,_,VLD,m)(VRD_T,int64_t,VID_T)



EXTERN_EFUNC(acos);
EXTERN_EFUNC(asin);
EXTERN_EFUNC(atan);
EXTERN_EFUNC(cos);
EXTERN_EFUNC(cosh);
EXTERN_EFUNC(exp);
EXTERN_EFUNC(log10);
EXTERN_EFUNC(log);
EXTERN_EFUNC(sin);
EXTERN_EFUNC(sinh);
EXTERN_EFUNC(tan);
EXTERN_EFUNC(tanh);

EXTERN_EFUNC2(atan2);
EXTERN_EFUNC2(mod);
EXTERN_EFUNC2(pow);

EXTERN_EFUNC2i(powi);
EXTERN_EFUNC2i1(powi1);

EXTERN_EFUNC2k(powk);
EXTERN_EFUNC2k1(powk1);


int32_t mask_sp[1<<VLS][VLS] __attribute__((aligned(64)));
int64_t mask_dp[1<<VLD][VLD] __attribute__((aligned(64)));
#if !defined(TARGET_WIN)
int32_t verbose = 0;
#else
#if VERBOSE == 0
int32_t verbose = 0;
#else
int32_t verbose = 1;
#endif
#endif


#if !defined(TARGET_WIN)
static void
parseargs(int argc, char *argv[])
{
	int opt;

	while ((opt = getopt(argc, argv, "v")) != -1) {
	    switch(opt) {
	    case 'v':
	        verbose = 1;
	        break;
	    default:
	        fprintf(stderr, "Usage %s [-v]\n",argv[0]);
	    }
	}
}
#endif


static VRS_T
vrs_set_arg(float fmin, float fconst )
{
    VRS_T   vret __attribute__((aligned(64)));
    float   fdelta;
    int     i;

    fdelta = fconst + fmin;
    for (i = 0; i < VLS; i++) {
       VRET(i) = (1.0f / (fdelta + (float) i));
    }

    return vret;
}

static VRD_T
vrd_set_arg(double dmin, double dconst )
{
    VRD_T   vret __attribute__((aligned(64)));
    double  ddelta;
    int     i;

    ddelta = dconst + dmin;
    for (i = 0; i < VLS; i++) {
       VRET(i) = (1.0 / (ddelta + (double) i));
    }

    return vret;
}


static void
build_masks(bool gray_code)
{
    int32_t    i;
    int32_t    j;
    int32_t    k;

    if (verbose) {
        printf("%s: %s mask vectors\n",
            __func__, gray_code ? "Gray code" : "binary");
    }

    memset(mask_sp, 0, sizeof mask_sp);
    memset(mask_dp, 0, sizeof mask_dp);


    for (j = 0; j < 1<<VLD; j++) {
        k = gray_code ? j ^ (j>>1) : j;
        for (i = 0; i < VLD; i++) {
            mask_dp[j][i] = (k&0x1) * -1;
            k = k>>1;
        }
    }

    for (j = 0; j < 1<<VLS; j++) {
        k = gray_code ? j ^ (j>>1) : j;
        for (i = 0; i < VLS; i++) {
            mask_sp[j][i] = (k&0x1) * -1;
            k = k>>1;
        }
    }

    if (verbose) {
    	for (j = 0; j < 1<<VLD; j++) {
        	for (i = 0; i < VLD; i++) {
            	printf(" %2lld", mask_dp[j][i]);
        	}
        	puts("");
    	}
    	for (j = 0; j < 1<<VLS; j++) {
        	for (i = 0; i < VLS; i++) {
            	printf(" %2d", mask_sp[j][i]);
        	}
        	puts("");
    	}
    }

    return;

}


static int
checkfltol1(float res, float exp, float ltol)
{
    int tests_passed = 0;
    int tests_failed = 0;

    if (exp == res) {
        tests_passed ++;
    }else if( exp != 0.0 && (fabsf((exp-res)/exp)) <= ltol ){
        tests_passed ++;
    }else if( exp == 0.0 && exp <= ltol ){
        tests_passed ++;
    } else {
        tests_failed ++;
	if (verbose) {
	    printf("test FAILED. res %f  exp %f\n", res, exp);
	}
    }

    if (verbose) {
	    if (tests_failed == 0) {
	        printf("1 test completed. %d tests PASSED. %d tests failed.\n",
	                      tests_passed, tests_failed);
	    } else {
	        printf("1 test completed. %d tests passed. %d tests FAILED.\n",
	                      tests_passed, tests_failed);
	    }
    }

    return(tests_failed);
}


static int
checkfltol(VRS_T res, VRS_T exp, VIS_T vvmask, int n, float ltol)
{
    int i;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
	if (VVMASK(i) != 0) {
	    tests_failed += checkfltol1(RES(i), EXP(i), ltol);
	}
    }

    return(tests_failed);
}

