/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * Math elemental function dispatch initialization.
 *
 * Dispatch initialization builds the runtime architecture specific dispatch
 * jump tables used by the generic elemental function compiler interface.
 *
 * There are two methods used to initialize the dispatch tables:
 * 1) C constructors - automatically call __math_dispatch() before main().
 *    The dispatch tables are undefined prior to calling __math_dispatch().
 *    This method is serialized by the linker support runtime library.
 *
 * 2) Construct on first elemental call.
 *    The dispatch table are fully defined (for all possible compiler
 *    generated entry points).  The definitions for each entry point are of
 *    the form __[frp][sdcz]_<NAME>_<VL>_init(), example:
 *    __fs_acos_1_init().
 *    Upon entry to __fs_acos_1_init, a call is made to __math_dispatch_init().
 *    __math_dispatch_init then only allows the "master" thread to enter
 *    __math_dispatch().  Any other non-master (not-first) thread will be held
 *    spin waiting for the local initialization flag "__math_dispatch_is_init"
 *    to to indicate __math_dispatch() has completed setup.
 *
 * Environment variable MTH_I_DEBUG is a bitmask used to control certain aspects of
 * dispatch initialization/shutdown.
 *
 * 0x1      Dump architecture specific dispatch table.
 * 0x2      Display MTH_I_OVERRIDE overrides
 * 0x4      Display MTH_I_OVERRIDE overrides - internal
 * 0x100    Init-on-first-call: one second delay for "master" thread in
 *          __math_dispatch_init() before entering __math_dispatch().
 *          Also generate stderr message for any subsequent "non-master"
 *          thread waiting for __math_dispatch() to complete.
 *
 * All debug/waring/error messages are directed to stderr.
 *
 */

#if     defined(TARGET_WIN)
/*
 * The Windows system header files are missing the argument list in the
 * following function declarations.  Without the argument list, albeit void,
 * dispatch.c cannot be compiled with the vectorcall ABI.
 *
 * Open Tools 10:
 *  I_RpcMgmtEnableDedicatedThreadPool
 * Visual Studio 2015:
 *  EnableMouseInPointerForThread
 *  GetThreadDpiHostingBehavior
 */

#define I_RpcMgmtEnableDedicatedThreadPool(...) \
        I_RpcMgmtEnableDedicatedThreadPool(void)
#define EnableMouseInPointerForThread(...)      \
        EnableMouseInPointerForThread(void)
#define GetThreadDpiHostingBehavior(...)        \
        GetThreadDpiHostingBehavior(void)
#endif

#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <string.h>
#include <time.h>
#include <inttypes.h>

#if defined(TARGET_WIN)
  #include <windows.h>
  #include <io.h>
  #define SLEEP(t) Sleep(t*1000)
  #define strcasecmp _stricmp
#endif

#if defined(TARGET_LINUX_X8664) || defined(TARGET_LINUX_POWER) || defined(TARGET_WIN)
#include <malloc.h>
#else
#include <sched.h>
#endif


#include "mth_tbldefs.h"

#if defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
#include "x86id.h"
#endif

#if     defined(TARGET_WIN)
#undef  I_RpcMgmtEnableDedicatedThreadPool
#undef  EnableMouseInPointerForThread
#undef  GetThreadDpiHostingBehavior
#endif

/*
 * To build for unit testing:
 *
 * gcc -E -P ~/pgi/dev/rte/pgc/hammer/src/math_tables/*.h | awk -v FS=,
 * 'function f(n){print "MTHTMPDEF("n")"}/^MTH/{f($3);f($4);f($5)}' |
 * sed 's/ //g' | sort | uniq | grep -v -e __math_dispatch_error >
 * ./tmp-mth_alldefs.h
 * awk \
 *     '/^MTH_DISPATCH_FUNC/ { \
 *       f = $1; \
 *       sub("^MTH_DISPATCH_FUNC\\(", "", f); \
 *       sub("\\).*", "", f); next; \
 *     } \
 *     /^[ \t]*_MTH_I_STATS_INC/ { \
 *       split($0, s, "[(,)]"); \
 *       print "MTH_DISPATCH_FUNC_STATS(" f "_prof, " s[2] \
 *       ", " s[3] ", ", s[4] ")"; f=""; \
 *     }' \
 *     ~/pgi/dev/rte/pgc//port/src/mth_128defs.c \
 * > ./tmp-mth_statsdefs.h
 *
 * gcc -I. -DUNIT_TEST=1 -DFOR_LIBPGC dispatch.c -Wall
 * ~/pgi/dev/rte/pgc/hammer/lib-linux86-64-pic/{cpu,x86}id.o -DTARGET_LINUX_X8664
 * -I/home/dparks/pgi/dev/rte/pgc/x86/src/ -DTARGET_LINUX_X8664
 * -I/home/dparks/pgi/dev/rte/pgc/hammer/src
 *
 * MTH_I_DEBUG=1 MTH_I_STATS=1 MTH_I_OVERRIDE=powi1+ss:f=p,powi1+ds:f=p ./a.out | grep powi1
 */

#define _STRINGIFY(s) #s
#define STRINGIFY(s) _STRINGIFY(s)

/*
 * Forward prototype definitions
 */
extern void __math_dispatch_error(void);
static char *fptr2char(void *);
static void __pgmath_abort(int, char *);

typedef p2f __mth_rt_vi_ptrs_t[func_size][sv_size][frp_size];

static char *carch[] = {
        /* List needs to follow arch_e in tbldefs.h */
#if defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
#define ARCH_DEFAULT arch_em64t
#define STR_ARCH_DEFAULT "em64t(p7)"
        [arch_em64t]    = "em64t",
        [arch_sse4]     = "sse4",
        [arch_avx]      = "avx",
        [arch_avxfma4]  = "avxfma4",
        [arch_avx2]     = "avx2",
        [arch_avx512knl]= "avx512knl",
        [arch_avx512]   = "avx512",
#elif   defined(TARGET_LINUX_POWER)
#define ARCH_DEFAULT arch_p8
#define STR_ARCH_DEFAULT "p8"
        [arch_p8]       = "p8",
        [arch_p9]       = "p9",
#elif   defined(TARGET_ARM64)
#define ARCH_DEFAULT arch_armv8
#define STR_ARCH_DEFAULT "armv8"
	[arch_armv8]    = "armv8",
	[arch_armv81a]  = "armv81a",
	[arch_armv82]   = "armv82",
#else
#define ARCH_DEFAULT arch_generic
#define STR_ARCH_DEFAULT "generic"
        [arch_generic]  = "generic",
#endif
};

static char *csv[] = {
        /* List needs to follow sv_e in mth_tbldefs.h */
        [sv_ss]         = "ss",
        [sv_ds]         = "ds",
        [sv_qs]         = "qs",
        [sv_cs]         = "cs",
        [sv_zs]         = "zs",
        [sv_cv1]        = "cv1",
        [sv_sv4]        = "sv4",
        [sv_dv2]        = "dv2",
        [sv_cv2]        = "cv2",
        [sv_zv1]        = "zv1",
        [sv_sv8]        = "sv8",
        [sv_dv4]        = "dv4",
        [sv_cv4]        = "cv4",
        [sv_zv2]        = "zv2",
        [sv_sv16]       = "sv16",
        [sv_dv8]        = "dv8",
        [sv_cv8]        = "cv8",
        [sv_zv4]        = "zv4",
        [sv_sv4m]       = "sv4m",
        [sv_dv2m]       = "dv2m",
        [sv_cv2m]       = "cv2m",
        [sv_zv1m]       = "zv1m",
        [sv_sv8m]       = "sv8m",
        [sv_dv4m]       = "dv4m",
        [sv_cv4m]       = "cv4m",
        [sv_zv2m]       = "zv2m",
        [sv_sv16m]      = "sv16m",
        [sv_dv8m]       = "dv8m",
        [sv_cv8m]       = "cv8m",
        [sv_zv4m]       = "zv4m",
};

static char *cfunc[] = {
        /* List needs to follow func_e in mth_tbldefs.h */
        [func_acos]     = "acos",
        [func_asin]     = "asin",
        [func_atan]     = "atan",
        [func_atan2]    = "atan2",
        [func_cos]      = "cos",
        [func_sin]      = "sin",
        [func_tan]      = "tan",
        [func_cosh]     = "cosh",
        [func_sinh]     = "sinh",
        [func_tanh]     = "tanh",
        [func_exp]      = "exp",
        [func_log]      = "log",
        [func_log10]    = "log10",
        [func_pow]      = "pow",
        [func_powi1]    = "powi1",
        [func_powi]     = "powi",
        [func_powk1]    = "powk1",
        [func_powk]     = "powk",
        [func_sincos]   = "sincos",
        [func_div]      = "div",
        [func_sqrt]     = "sqrt",
        [func_mod]      = "mod",
        [func_aint]     = "aint",
        [func_ceil]     = "ceil",
        [func_floor]    = "floor",
};

#undef SLEEF
#include "math_tables/mth_acosdefs.h"
#include "math_tables/mth_asindefs.h"
#include "math_tables/mth_atandefs.h"
#include "math_tables/mth_atan2defs.h"
#include "math_tables/mth_cosdefs.h"
#include "math_tables/mth_sindefs.h"
#include "math_tables/mth_tandefs.h"
#include "math_tables/mth_coshdefs.h"
#include "math_tables/mth_sinhdefs.h"
#include "math_tables/mth_tanhdefs.h"
#include "math_tables/mth_expdefs.h"
#include "math_tables/mth_logdefs.h"
#include "math_tables/mth_log10defs.h"
#include "math_tables/mth_powdefs.h"
#include "math_tables/mth_powidefs.h"
#include "math_tables/mth_sincosdefs.h"
#include "math_tables/mth_divdefs.h"
#include "math_tables/mth_sqrtdefs.h"
#include "math_tables/mth_moddefs.h"
#include "math_tables/mth_aintdefs.h"
#include "math_tables/mth_ceildefs.h"
#include "math_tables/mth_floordefs.h"
#ifdef SLEEF
#include "math_tables/mth_sleef.h"
#endif
#undef  DO_MTH_DISPATCH_FUNC
#define DO_MTH_DISPATCH_FUNC(name_, func_, sv_, frp_) extern void name_##_init(void);
#include "tmp-mth_statsdefs.h"
#undef  DO_MTH_DISPATCH_FUNC
#define DO_MTH_DISPATCH_FUNC(name_, func_, sv_, frp_) extern void name_##_prof(void);
#include "tmp-mth_statsdefs.h"

#undef MTHINTRIN
#define MTHINTRIN(_func, _sv, _a, _f, _r, _p, _s)                      \
  {                                                                    \
    .arch = arch_##_a, .func = func_##_func, .sv = sv_##_sv, .pf = _f, \
    .pr = _r, .pp = _p, .ps=_s,                                        \
  }                                                                    \
  ,

static mth_intrins_defs_t mth_intrins_defs[] = {
#ifndef SLEEF
#include "math_tables/mth_acosdefs.h"
#include "math_tables/mth_asindefs.h"
#include "math_tables/mth_atandefs.h"
#include "math_tables/mth_atan2defs.h"
#include "math_tables/mth_cosdefs.h"
#include "math_tables/mth_sindefs.h"
#include "math_tables/mth_tandefs.h"
#include "math_tables/mth_coshdefs.h"
#include "math_tables/mth_sinhdefs.h"
#include "math_tables/mth_tanhdefs.h"
#include "math_tables/mth_expdefs.h"
#include "math_tables/mth_logdefs.h"
#include "math_tables/mth_log10defs.h"
#include "math_tables/mth_powdefs.h"
#include "math_tables/mth_powidefs.h"
#include "math_tables/mth_sincosdefs.h"
#include "math_tables/mth_divdefs.h"
#include "math_tables/mth_sqrtdefs.h"
#include "math_tables/mth_moddefs.h"
#include "math_tables/mth_aintdefs.h"
#include "math_tables/mth_ceildefs.h"
#include "math_tables/mth_floordefs.h"
#else
#include "math_tables/mth_sleef.h"
#endif
};

#undef  DO_MTH_DISPATCH_FUNC
#define DO_MTH_DISPATCH_FUNC(name_, func_, sv_, frp_) \
    [func_][sv_][frp_] = name_##_prof,

static __mth_rt_vi_ptrs_t __mth_rt_vi_ptrs_statdefs = {
#include "tmp-mth_statsdefs.h"
};

/*
 * __mth_rt_vi_ptrs and __mth_rt_vi_ptrs_stat are global tables referenced
 * by __[frp]_<NAME>_[VL] routines.
 */

#undef  DO_MTH_DISPATCH_FUNC
#define DO_MTH_DISPATCH_FUNC(name_, func_, sv_, frp_) \
    [func_][sv_][frp_] = name_##_init,

__mth_rt_vi_ptrs_t __mth_rt_vi_ptrs = {
#include "tmp-mth_statsdefs.h"
};

__mth_rt_vi_ptrs_t __mth_rt_vi_ptrs_stat;
uint64_t __mth_rt_stats[frp_size][func_size][sv_size];
static  __mth_rt_vi_ptrs_t __mth_rt_vi_ptrs_new;

#if !defined(_WIN64) && !defined(DISPATCH_IS_STATIC)
#define CONSTRUCTOR __attribute__((constructor(101)))
#define DESTRUCTOR __attribute__((destructor))
#else
#define CONSTRUCTOR
#define DESTRUCTOR
#endif

static volatile bool    __math_dispatch_is_init = false;    // Local flag
static          bool    __math_dispatch_in_prog = false;    // Local flag

static arch_e __math_target = arch_size; /* No known processor */

typedef struct {
  arch_e parch; /* Processor architecture */
  char *pname;
} text2archtype_t;

static text2archtype_t text2archtype[] = {
#if defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
        {arch_em64t,    "p7"},
        {arch_sse4,     "core2"},
        {arch_sse4,     "penryn"},
        {arch_sse4,     "nehalem"},
        {arch_avx,      "sandybridge"},
        {arch_em64t,    "k8"},
        {arch_em64t,    "k8e"},
        {arch_sse4,     "barcelona"},
        {arch_sse4,     "shanghai"},
        {arch_sse4,     "istanbul"},
        {arch_avxfma4,  "bulldozer"},
        {arch_avxfma4,  "piledriver"},
        {arch_avx2,     "haswell"},
        {arch_avx512knl,"knightslanding"},
        {arch_avx512,   "skylake"},

        {arch_em64t,    "em64t"},
        {arch_sse4,     "sse4"},
        {arch_avx,      "avx"},
        {arch_avxfma4,  "avxfma4"},
        {arch_avx2,     "avx2"},
        {arch_avx512knl,"avx512knl"},
        {arch_avx512,   "avx512"},
#endif
#ifdef TARGET_LINUX_POWER
        {arch_p8,       "p8"},
        {arch_p8,       "pwr8"},
        {arch_p9,       "p9"},
        {arch_p9,       "pwr9"},
#endif
#ifdef TARGET_ARM64
	{arch_armv8,    "armv8"},
	{arch_armv81a,  "armv81a"},
	{arch_armv82,    "armv82"},
#endif
#ifdef TARGET_LINUX_GENERIC
        {arch_generic,  "generic"},
#endif
};

typedef struct  {
    elmtsz_e   elmtsz;    // Size of elements
    int         nelmt;     // Number of elements
} sv2attributes_t;

static  sv2attributes_t sv2attributes[] = {
        [sv_ss]     = { .elmtsz=elmtsz_32,     .nelmt=1},
        [sv_cs]     = { .elmtsz=elmtsz_32,     .nelmt=1},
        [sv_cv1]    = { .elmtsz=elmtsz_32,     .nelmt=1},

        [sv_ds]     = { .elmtsz=elmtsz_64,     .nelmt=1},
        [sv_zs]     = { .elmtsz=elmtsz_64,     .nelmt=1},

        [sv_qs]     = { .elmtsz=elmtsz_128,    .nelmt=1},

        [sv_sv4]    = { .elmtsz=elmtsz_128,    .nelmt=4*1},
        [sv_dv2]    = { .elmtsz=elmtsz_128,    .nelmt=2*1},
        [sv_cv2]    = { .elmtsz=elmtsz_128,    .nelmt=2*1},
        [sv_zv1]    = { .elmtsz=elmtsz_128,    .nelmt=1*1},
        [sv_sv4m]   = { .elmtsz=elmtsz_128,    .nelmt=4*1},
        [sv_dv2m]   = { .elmtsz=elmtsz_128,    .nelmt=2*1},
        [sv_cv2m]   = { .elmtsz=elmtsz_128,    .nelmt=2*1},
        [sv_zv1m]   = { .elmtsz=elmtsz_128,    .nelmt=1*1},

        [sv_sv8]    = { .elmtsz=elmtsz_256,    .nelmt=4*2},
        [sv_dv4]    = { .elmtsz=elmtsz_256,    .nelmt=2*2},
        [sv_cv4]    = { .elmtsz=elmtsz_256,    .nelmt=2*2},
        [sv_zv2]    = { .elmtsz=elmtsz_256,    .nelmt=1*2},
        [sv_sv8m]   = { .elmtsz=elmtsz_256,    .nelmt=4*2},
        [sv_dv4m]   = { .elmtsz=elmtsz_256,    .nelmt=2*2},
        [sv_cv4m]   = { .elmtsz=elmtsz_256,    .nelmt=2*2},
        [sv_zv2m]   = { .elmtsz=elmtsz_256,    .nelmt=1*2},

        [sv_sv16]   = { .elmtsz=elmtsz_512,    .nelmt=4*4},
        [sv_dv8]    = { .elmtsz=elmtsz_512,    .nelmt=2*4},
        [sv_cv8]    = { .elmtsz=elmtsz_512,    .nelmt=2*4},
        [sv_zv4]    = { .elmtsz=elmtsz_512,    .nelmt=1*4},
        [sv_sv16m]  = { .elmtsz=elmtsz_512,    .nelmt=4*4},
        [sv_dv8m]   = { .elmtsz=elmtsz_512,    .nelmt=2*4},
        [sv_cv8m]   = { .elmtsz=elmtsz_512,    .nelmt=2*4},
        [sv_zv4m]   = { .elmtsz=elmtsz_512,    .nelmt=1*4},
};

static char *elmtsz2text[] = {
        [elmtsz_32]     = "32",
        [elmtsz_64]     = "64",
        [elmtsz_128]    = "128",
        [elmtsz_256]    = "256",
        [elmtsz_512]    = "512",
};

static frp_e __mth_fast = frp_f;
static frp_e __mth_relaxed = frp_r;
static frp_e __mth_precise = frp_p;
static frp_e __mth_sleef = frp_s;

static char *frp2text[] = { // table is ordered by frp_e values
        [frp_f] = "fast",
        [frp_r] = "relaxed",
        [frp_p] = "precise",
        [frp_s] = "sleef",
};

/*
 * Table of MTH_I_{FAST,RELAXED,PRECISE} environment variables and
 * their default values.
 */

typedef struct {
  char *var;    // Environment variable name
  frp_e *val;   // Pointer to variable containing setting
  frp_e defval; // Default value
} frp_env_t;

static frp_env_t frp_env[] = {
    {.var = "MTH_I_FAST",    .val = &__mth_fast, .defval = frp_f},
    {.var = "MTH_I_RELAXED", .val = &__mth_relaxed, .defval = frp_r},
    {.var = "MTH_I_PRECISE", .val = &__mth_precise, .defval = frp_p},
    {.var = "MTH_I_SLEEF",   .val = &__mth_precise, .defval = frp_s},
};

/*
 * A quick note on the conventions followed for output from
 * the dispatch initialization/termination.  All debug messages are
 * printed to standard out.  Debug messages are enabled when
 * environment variable MTH_I_DEBUG is nonzero.  Though different levels
 * of debug messages can be printed with specific settings of MTH_I_DEBUG.
 * All error messages are directed to standard error.
 * Might not be the best convention, but it is consistent.
 */

static uint64_t __mth_i_debug = 0;

typedef enum    {
    stats_none      = 0x0,
    stats_summary   = 1<<0,         // Summary by intrinsic type
    stats_by_type   = 1<<1,         // By type, scalar, vector
    stats_by_func   = 1<<2,         // Individual
    stats_disp_err  = 1<<3,         // Dispatch error (internal)
    stats_all       = stats_summary | stats_by_type | stats_by_func,
} stats_e;
static stats_e __mth_i_stats = stats_none;

typedef struct {
  void *pfunc; // Pointer to function
  char *cname; // Function's name
} fptr2name_t;

/*
 * Table fptr2name is used for debugging.
 */

#undef MTHTMPDEF
#define MTHTMPDEF(_f)         \
  {                           \
    .pfunc = _f, .cname = #_f \
  }                           \
  ,

static fptr2name_t fptr2name[] = {
    {.pfunc = __math_dispatch_error, .cname = "__math_dispatch_error"},
#include "tmp-mth_alldefs.h"
};

static int
get_string_index(char *s, char *tbl[], int len)
{
  int i;

  for (i = 0; i < len; i++) {
    if (strcmp(s, tbl[i]) == 0) {
      return i;
    }
  }

  return -1;
}

static bool
is_frp_char_valid(char c)
{

  return (c == 'f' || c == 'r' || c == 'p' || c == 's');
}

/*
 * frp2index() - return enumerated type given frp in character
 * class [frps].
 *
 * Assumes that character has been previously validated.
 */

static frp_e
frp2index(char frp)
{
  switch (frp) {
  case 'f':
    return frp_f;
  case 'r':
    return frp_r;
  case 'p':
    return frp_p;
  case 's':
    return frp_s;
  }
  return frp_f; // Added to get rid of compiler warnings
}

static void
mth_i_override_usage()
{
  int i;
  fprintf(
      stderr,
      "Usage: MTH_I_OVERRIDE=<INTRIN_NAME>+<SV>:<ORIG_FRPS>=<ALT_FRPS>[,...]\n"
      "Where:\n"
      "\t<INTRIN_NAME> is one of:");

  for (i = 0; i < sizeof cfunc / sizeof *cfunc; i++) {
    fprintf(stderr, " %s", cfunc[i]);
  }

  fprintf(stderr, "\n\t<SV> is one of:");

  for (i = 0; i < sizeof csv / sizeof *csv; i++) {
    fprintf(stderr, " %s", csv[i]);
  }

  fputs("\n\t<ORIG_FRP> (original) is one of the character set \"F|R|P|S\"\n"
        "\t<ALT_FRP>  (alternative) is one of the character set \"F|R|P|S\"\n", stderr);
}

static void
mth_i_override()
{
  char *penv;
  char *tbuf;
  char *name;
  char *sv;
  char corigfrp;
  char caltfrp;
  int i;
  int j;
  sv_e s;
  frp_e origfrp;
  frp_e altfrp;
  func_e f;
  bool failure;

  penv = getenv("MTH_I_OVERRIDE");
  if (__mth_i_debug & 0x2) {
    fprintf(stderr, "MTH_I_OVERRIDE=%s\n", penv);
  }
  if (penv == NULL) {
    return;
  }

  j = strlen(penv) + 1;
  tbuf = malloc(j);
  for (i = 0; i < j; i++) {
    tbuf[i] = tolower(penv[i]);
  }

  /*
   * Real simple state machine.
   * MTH_I_OVERRIDE=<INTRIN_NAME>+<SV>:<ORIG_FRP>=<ALT_FRP>[,...]
   * Where:
   *    <INTRIN_NAME> is a known intrinsic name in the dispatch table.
   *    <SV> is the scalar/vector size designator/descriptor.
   *    <ORIG_FRP> (original) is one of the character set "F|R|P"
   *    <ALT_FRP>  (alternative) is one of the character set "F|R|P"
   *
   * State machine does not ignore whitespace.
   *
   * Valid examples:
   * MTH_I_OVERRIDE=acos+sd:f=p
   * MTH_I_OVERRIDE=acos+sd:f=p,acos+vs4:r=f
   * MTH_I_OVERRIDE=acos+sd:f=p,acos+vs4:r=f,
   *
   * Invalid examples:
   * MTH_I_OVERRIDE=acos-sd:f=p
   * MTH_I_OVERRIDE=acos-sd;f=p
   * MTH_I_OVERRIDE=acos+sd;f-p
   * MTH_I_OVERRIDE=acos-sd;f,
   */

  failure = true;
  i = 0;
  while (true) {
    name = NULL;
    sv = NULL;

    /*
     * <INTRIN_NAME>
     */
    if (isalpha(tbuf[i]) == 0) {
      // fprintf(stderr, "tbuf[%d]=%c must start with a letter\n", i, tbuf[i]);
      break;
    }
    name = &tbuf[i];
    for (i++; isalnum(tbuf[i]) != 0; i++) {
    }

    /*
     * Seperator "+".
     */
    if (tbuf[i] != '+') {
      // fprintf(stderr, "tbuf[%d]=%c must be a '+'\n", i, tbuf[i]);
      break;
    }

    /*
     * <SV>
     */
    tbuf[i++] = '\0';
    sv = &tbuf[i];
    for (i++; isalnum(tbuf[i]) != 0; i++) {
    }

    /*
     * Seperator ":".
     */
    if (tbuf[i] != ':') {
      // fprintf(stderr, "tbuf[%d]=%c must be a ':'\n", i, tbuf[i]);
      break;
    }

    /*
     * <ORIG_FRP>
     */
    tbuf[i++] = '\0';
    if (isalpha(tbuf[i])) {
      corigfrp = tbuf[i++];
    } else {
      // fprintf(stderr, "tbuf[%d]=%c must be one of 'frp'\n", i, tbuf[i]);
      break;
    }

    /*
     * Seperator "=".
     */
    if (tbuf[i++] != '=') {
      // fprintf(stderr, "tbuf[%d]=%c must be a '='\n", i, tbuf[i]);
      break;
    }

    /*
     * <ALT_FRP>
     */
    caltfrp = tbuf[i++];

    /*
     * End of buffer or seperator "," for more intrinsics.
     */
    if (tbuf[i] != '\0' && tbuf[i] != ',') {
      break;
    }

    j = get_string_index(name, cfunc, sizeof cfunc / sizeof *cfunc);
    if (j < 0) {
      fprintf(stderr, "MTH_I_OVERRIDE: intrinsic name %s not valid\n", name);
      break;
    }
    f = j;

    j = get_string_index(sv, csv, sizeof csv / sizeof *csv);
    if (j < 0) {
      fprintf(stderr, "MTH_I_OVERRIDE: intrinsic class %s not valid\n", sv);
      break;
    }
    s = j;

    if (false == is_frp_char_valid(corigfrp)) {
      fprintf(stderr,
              "MTH_I_OVERRIDE: Original precision=%c must be one of 'frps'\n",
              corigfrp);
      break;
    }

    if (false == is_frp_char_valid(caltfrp)) {
      fprintf(stderr,
              "MTH_I_OVERRIDE: Alternative precision=%c must be one of 'frps'\n",
              caltfrp);
      break;
    }

    origfrp = frp2index(corigfrp);
    altfrp = frp2index(caltfrp);

    if (__mth_i_debug & (0x2 | 0x4)) {
      if (__mth_i_debug & 0x4) {
        fprintf(stderr, "MTH_I_OVERRIDE: name=%s,sv=%s,old=%c,new=%c\n", name,
                sv, corigfrp, caltfrp);
        fprintf(stderr, "MTH_I_OVERRIDE: f=%d, s=%d, iorigfrp=%d, ialtfrp=%d\n",
                f, s, origfrp, altfrp);
      }
      fprintf(stderr, "MTH_I_OVERRIDE: Replacing %s(%s):%c=%s with %c=%s\n",
              name, sv, corigfrp, fptr2char(__mth_rt_vi_ptrs_new[f][s][origfrp]),
              caltfrp, fptr2char(__mth_rt_vi_ptrs_new[f][s][altfrp]));
    }

    /*
     * Override dispatch entry.
     */
    __mth_rt_vi_ptrs_new[f][s][origfrp] = __mth_rt_vi_ptrs_new[f][s][altfrp];

    if (tbuf[i] == '\0' || (tbuf[i] == ',' && tbuf[i + 1] == '\0')) {
      failure = false;
      break;
    }
    tbuf[i++] = '\0';
  }

  if (true == failure) {
    mth_i_override_usage();
    fprintf(stderr, "Errors in processing MTH_I_OVERRIDE");
    exit(1);
  }

  free(tbuf);
}

static char *
fptr2char(void *fptr)
{
  int i;

  for (i = 0; i < sizeof(fptr2name) / sizeof(*fptr2name); i++) {
    if (fptr2name[i].pfunc == fptr) {
      return (fptr2name[i].cname);
    }
  }
  return ("MISSING NAME");
}

static void
dump_mth_rt_vi_ptrs()
{
  func_e f;
  sv_e s;
  int lcsv = sizeof(csv) / sizeof(*csv); // Length of string structure csv
  __mth_rt_vi_ptrs_t *p = &__mth_rt_vi_ptrs;
  if (__mth_i_stats != 0) {
    p = &__mth_rt_vi_ptrs_stat;
  }

  for (f = 0; f < func_size; f++) {
    for (s = 0; s < lcsv; s++) { // Note sv_size can/is larger than csv
     fprintf(stderr,
             "%-5s %-5s %p %-21s %p %-21s %p %-21s %p %-21s\n", cfunc[f], csv[s],
             (*p)[f][s][frp_f], fptr2char((*p)[f][s][frp_f]),
             (*p)[f][s][frp_r], fptr2char((*p)[f][s][frp_r]),
             (*p)[f][s][frp_p], fptr2char((*p)[f][s][frp_p]),
             (*p)[f][s][frp_s], fptr2char((*p)[f][s][frp_s]));
    }
  }
}


static void
__math_epilog_do_stats()
{
  func_e f;
  sv_e s;
  frp_e frp;
  elmtsz_e elmtsz;
  int lcsv = sizeof(csv) / sizeof(*csv); // Length of string structure csv
  uint64_t totcalls;
  uint64_t callsbyfunc[func_size];
  uint64_t elmtbysz[elmtsz_size];
  uint64_t totelmt;
  uint64_t t;
  char  *pf;
  char  *pi;

  totelmt = totcalls = 0;
  memset(elmtbysz, 0, sizeof elmtbysz);
  memset(callsbyfunc, 0, sizeof callsbyfunc);

  for (f = 0; f < func_size; f++) {
    uint64_t t1;
    t = 0;
    for (s = 0; s < lcsv; s++) { // Note sv_size can/is larger than csv
      t1 = __mth_rt_stats[frp_f][f][s] + __mth_rt_stats[frp_r][f][s] +
         __mth_rt_stats[frp_p][f][s];
      t += t1;
      elmtbysz[sv2attributes[s].elmtsz] += t1;
      totelmt += t1*sv2attributes[s].nelmt;
    }
    totcalls += t;
    callsbyfunc[f] = t;
  }

  if (0 != (stats_disp_err & __mth_i_stats)) {
    /*
     * Internal only - __math_dispatch_error() was invoked.
     * Identify which intrinsic call is not mapped to this architecture.
     */
    for (f = 0; f < func_size; f++) {
      for (s = 0; s < lcsv; s++) { // Note sv_size can/is larger than csv
        for (frp = 0; frp < frp_size; frp++) {
          if (__mth_rt_stats[frp][f][s] != 0 &&
             __math_dispatch_error == __mth_rt_vi_ptrs_stat[f][s][frp]) {
            fprintf(stderr,
              "****\t%s/%s/%s\t****"
              "Entry point not defined for CPU target=%s.\n",
              cfunc[f], csv[s], frp2text[frp], carch[__math_target]);
          }
        }
      }
    }
  }
  /*
   * Allow the user to specify all three classes of statistics to be displayed.
   */
  if (0 != (stats_summary & __mth_i_stats)) {
    fputs(
      "\n"
      "\t\tIntrinsic Summary by Name\n"
      "\t\t--------- ------- -- ----\n"
      "INTRIN\t     #calls    %tot\n"
      , stderr);
    for (f = 0; f < func_size; f++) {
      if (callsbyfunc[f] != 0) {
        fprintf(stderr, "%-6s %12" PRIu64 " %6.2f%%\n",
          cfunc[f], callsbyfunc[f], 100.0*callsbyfunc[f]/totcalls);
      }
    }
  }

  if (0 != (stats_by_type & __mth_i_stats)) {
    fputs(
      "\n"
      "\t\tIntrinsic Summary by Type\n"
      "\t\t--------- ------- -- ----\n"
      "INTRIN\tTYPE\t    #calls    %tot    #elements    %tot\n"
      , stderr);
    for (f = 0; f < func_size; f++) {
      if (callsbyfunc[f] != 0) {
        pf = cfunc[f];
        for (s = 0; s < lcsv; s++) { // Note sv_size can/is larger than csv
          uint64_t telmt;
          t = __mth_rt_stats[frp_f][f][s] + __mth_rt_stats[frp_r][f][s] +
               __mth_rt_stats[frp_p][f][s];
          if (t != 0) {
            telmt = t * sv2attributes[s].nelmt;
            fprintf(stderr, "%-6s\t%-5s %12" PRIu64 " %6.2f%% %12" PRIu64 " %6.2f%%\n",
              pf, csv[s], t, 100.0*t/totcalls, telmt, 100.0*telmt/totelmt);
          pf = "";
          }
        }
      }
    }
  }

  if (0 != (stats_by_func & __mth_i_stats)) {
    fputs(
      "\n"
      "\t\tIntrinsic Summary by Entry Point\n"
      "\t\t--------- ------- -- ----- -----\n\n"
      "INTRIN\tTYPE\tENTRY PT\t\t   #calls    %tot    #elements    %tot\n"
      , stderr);
    for (f = 0; f < func_size; f++) {
      if (callsbyfunc[f] != 0) {
        pf = cfunc[f];
        for (s = 0; s < lcsv; s++) { // Note sv_size can/is larger than csv
          uint64_t telmt;
          pi = csv[s];
          t = __mth_rt_stats[frp_f][f][s] + __mth_rt_stats[frp_r][f][s] +
               __mth_rt_stats[frp_p][f][s];
          if (t != 0) {
            telmt = t * sv2attributes[s].nelmt;
            for (frp = 0; frp < frp_size; frp++) {
              if (__mth_rt_stats[frp][f][s] != 0) {
                telmt = __mth_rt_stats[frp][f][s] * sv2attributes[s].nelmt;
                fprintf(stderr, "%-6s\t%-5s\t%-20s %12" PRIu64 " %6.2f%% %12" PRIu64" %6.2f%%\n",
                  pf, pi, fptr2char(__mth_rt_vi_ptrs_stat[f][s][frp]),
                  __mth_rt_stats[frp][f][s],
                  100.0*__mth_rt_stats[frp][f][s]/totcalls,
                  telmt, 100.0*telmt/totelmt);
              }
            }
            pi = "";
            pf = "";
          }
        }
      }
    }
  }


  fprintf(stderr, "\n\nTotal calls:\t%12" PRIu64 "\n", totcalls);
  fprintf(stderr, "Total elements:\t%12" PRIu64 "\n", totelmt);

  fputs("\nTotal number of calls by element size\n"
          "-------------------------------------\n", stderr);
  for(elmtsz = 0; elmtsz < elmtsz_size; elmtsz++) {
    fprintf(stderr, "%3s:\t%12" PRIu64 "\n", elmtsz2text[elmtsz], elmtbysz[elmtsz]);
  }
  if (0 != ((stats_by_type | stats_by_func) & __mth_i_stats)) {
    fputs(
      "\n"
      "\t\tIntrinsic Type Legend\n"
      "\t\t--------- ---- ------\n\n"
      "ss\t32-bit real scalar\t\tds\t64-bit real scalar\n"
      "cs\t32-bit complex scalar\t\tzs\t64-bit complex scalar\n"
      "cv1\t32-bit complex(packed) scalar\n"
      "sv4\t4*32-bit real vector\t\tdv2\t2*64-bit real vector\n"
      "cv2\t2*32-bit complex vector\t\tzv1\t2*64-bit complex vector(packed)\n"
      "sv8\t8*32-bit real vector\t\tdv4\t4*64-bit real vector\n"
      "cv4\t4*32-bit complex vector\t\tzv2\t2*64-bit complex vector\n"
      "sv16\t16*32-bit real vector\t\tdv8\t8*64-bit real vector\n"
      "cv8\t8*32-bit complex vector\t\tzv4\t4*64-bit complex vector\n"
      , stderr);
  }
  fflush(stderr);
}

/*
 * __math_epilog_()
 */

void DESTRUCTOR
__math_epilog_()
{
  if (__mth_i_stats != 0) {
    __math_epilog_do_stats();
  }
}


/*
 * __math_dispatch()
 */

void CONSTRUCTOR
__math_dispatch()
{

  char *ptenv;
  int i;
  int j;
  func_e f;
  mth_intrins_defs_t *p;
  sv_e s;

  if (NULL != (ptenv = getenv("MTH_I_DEBUG"))) {
    char *pend;
    __mth_i_debug = strtol(ptenv, &pend, 0);
    if (*ptenv != '\0' && *pend == '\0') {
      fprintf(stderr, "MTH_I_DEBUG=%" PRIu64 "\n", __mth_i_debug);
    } else {
      fprintf(stderr, "MTH_I_DEBUG=%s not a valid number - disabled\n", ptenv);
      __mth_i_debug = 0;
    }
  }

  if (NULL != (ptenv = getenv("MTH_I_ARCH"))) {
    for (i = 0; i < sizeof(text2archtype) / sizeof(*text2archtype); i++) {
      if (0 == strcasecmp(text2archtype[i].pname, ptenv)) {
        __math_target = text2archtype[i].parch;
        break;
      }
    }
    if (__math_target == arch_size) {
      fputs("The only valid values for MTH_I_ARCH are:\n", stderr);
      for (i = 0; i < sizeof(text2archtype) / sizeof(*text2archtype); i++) {
        fprintf(stderr, " %s", text2archtype[i].pname);
      }
      fprintf(stderr, "\nDefaulting to " STR_ARCH_DEFAULT "\n");
      __math_target = ARCH_DEFAULT;
    }

  } else if (NULL != (ptenv = getenv("PGI_FASTMATH_CPU"))) {
    for (i = 0; i < sizeof(text2archtype) / sizeof(*text2archtype); i++) {
      if (0 == strcasecmp(text2archtype[i].pname, ptenv)) {
        __math_target = text2archtype[i].parch;
        break;
      }
    }
    if (__math_target == arch_size) {
      fputs("The only valid values for MTH_I_ARCH are:\n", stderr);
      for (i = 0; i < sizeof(text2archtype) / sizeof(*text2archtype); i++) {
        fprintf(stderr, " %s", text2archtype[i].pname);
      }
      fprintf(stderr, "\nDefaulting to " STR_ARCH_DEFAULT "\n");
      __math_target = ARCH_DEFAULT;
    }

  } else { /* Get processor architecture using CPUID information */
#if defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_WIN_X8664)
    if (X86IDFN(is_avx512vl)() == 1) {
      __math_target = arch_avx512;
    } else if (X86IDFN(is_avx512f)() == 1) {
      __math_target = arch_avx512knl;
    } else if (X86IDFN(is_avx2)() == 1) {
      __math_target = arch_avx2;
    } else if (X86IDFN(is_avx)() == 1) {
      if (X86IDFN(is_intel)() == 1) {
        __math_target = arch_avx;
      }
      if (X86IDFN(is_amd)() == 1) {
        if (X86IDFN(is_fma4)() == 1) {
          __math_target = arch_avxfma4;
        } else {
          __math_target = arch_sse4;
        }
      }
    } else {
      if ((X86IDFN(is_sse4a)() == 1) || (X86IDFN(is_sse41)() == 1)) {
        __math_target = arch_sse4;
      } else {
        __math_target = arch_em64t;
      }
    }
#endif
#ifdef TARGET_LINUX_POWER
    __math_target = ARCH_DEFAULT;
#endif
#ifdef TARGET_ARM64
    __math_target = ARCH_DEFAULT;
#endif
#ifdef TARGET_LINUX_GENERIC
    __math_target = ARCH_DEFAULT;
#endif
  }

  /*  Allow overriding of the default functions called for fast,
   *  relaxed, and precise intrinsics.   This is done by setting
   *  the following environment variables:
   *
   *        MTH_I_FAST    = {fast,relaxed,precise,sleef}
   *        MTH_I_RELAXED = {fast,relaxed,precise,sleef}
   *        MTH_I_PRECISE = {fast,relaxed,precise,sleef}
   *        MTH_I_SLEEF   = {fast,relaxed,precise,sleef}
   *
   */

  for (j = 0; j < sizeof(frp_env) / sizeof(*frp_env); j++) {
    if (NULL != (ptenv = getenv(frp_env[j].var))) {
      *frp_env[j].val = frp_size;
      for (i = 0; i < sizeof(frp2text) / sizeof(*frp2text); i++) {
        if (0 == strcasecmp(frp2text[i], ptenv)) {
          *frp_env[j].val = i;
          break;
        }
      }
      if (*frp_env[j].val == frp_size) {
        fprintf(stderr, "The only valid values for %s are:\n", frp_env[j].var);
        for (i = 0; i < sizeof(frp2text) / sizeof(*frp2text); i++) {
          fprintf(stderr, " %s", frp2text[i]);
        }
        *frp_env[j].val = frp_env[j].defval;
        fprintf(stderr, "\nDefaulting to %s\n", frp2text[*frp_env[j].val]);
      }
    }
  }

  if (__mth_i_debug & 0x1) {
    fprintf(stderr, "__math_target: %s(%d)\n", carch[__math_target],
            __math_target);
    fprintf(stderr, "__mth_fast: %s(%d)\n", frp2text[__mth_fast], __mth_fast);
    fprintf(stderr, "__mth_relaxed: %s(%d)\n", frp2text[__mth_relaxed],
            __mth_relaxed);
    fprintf(stderr, "__mth_precise: %s(%d)\n", frp2text[__mth_precise],
            __mth_precise);
    fprintf(stderr, "__mth_sleef: %s(%d)\n", frp2text[__mth_sleef],
            __mth_sleef);
    fputs("__math_dispatch: built on " __DATE__ " " __TIME__ " with "
#if     defined(__clang__)
           // Too much internal info "clang-" __clang_version__
           "clang-" STRINGIFY(__clang_major__ )
                "." STRINGIFY(__clang_minor__)
                "." STRINGIFY(__clang_patchlevel__)
#elif   defined(__GNUC__)
           "gcc-" STRINGIFY(__GNUC__)
              "." STRINGIFY(__GNUC_MINOR__)
              "." STRINGIFY(__GNUC_PATCHLEVEL__)
#elif   defined(__PGIC__)
           "pgcc-" STRINGIFY(__PGIC__)
               "." STRINGIFY(__PGIC_MINOR__)
               "." STRINGIFY(__PGIC_PATCHLEVEL__)
#else
           "Unknown build compiler"
#endif
           "\n", stderr);
  }

  /*
   * 2 pass scan of table definitions.
   * 1st pass:
   *    populate __mth_rt_vi_ptrs with all mth_intrins_defs entries that
   *    have arch_any.
   * 2nd pass:
   *    populate __mth_rt_vi_ptrs with entries whose architecture match
   *    __math_target.
   *
   * We need two passes to build the dispatch table because there are not
   * ordering rules for where arch_any can be specified in the definition
   * files.
   *
   * Probably not the most efficient way to scan the definition table twice,
   * but the table is relatively speaking not that large.
   */

  for (j = 0; j < 2; j++) {
    arch_e target_arch;

    target_arch = j == 0 ? arch_any : __math_target;
    for (i = 0; i < sizeof(mth_intrins_defs) / sizeof(*mth_intrins_defs); i++) {
      p = &mth_intrins_defs[i];

      /*
       * XXX - testing p->arch == __math_target is not sufficient,
       * since arch_any is 0, and all the undefined entries will have
       * arch == 0.
       * But defined entries will have one or more valid pointers to intrinsic
       * functions.
       *
       * Macro MTHINTRIN requires all 3 F/R/P fields to be defined, though
       * technically nothing prevents any one of those fields from using
       * "NULL".
       *
       * Here to check for a valid definition we only check the first field
       * F(ast) for a non NULL value.
       */

      if (p->arch == target_arch && p->pf != NULL) {
        if (__mth_fast == frp_f) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_f] = p->pf;
        } else if (__mth_fast == frp_r) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_f] = p->pr;
        } else if (__mth_fast == frp_p) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_f] = p->pp;
        } else {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_f] = p->ps;
        }

        if (__mth_relaxed == frp_f) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_r] = p->pf;
        } else if (__mth_relaxed == frp_r) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_r] = p->pr;
        } else if (__mth_relaxed == frp_p) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_r] = p->pp;
        } else {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_r] = p->ps;
        }

        if (__mth_precise == frp_f) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_p] = p->pf;
        } else if (__mth_precise == frp_r) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_p] = p->pr;
        } else if (__mth_precise == frp_p) {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_p] = p->pp;
        } else {
          __mth_rt_vi_ptrs_new[p->func][p->sv][frp_p] = p->ps;
        }
      }
    }
  }

  mth_i_override();

  /*
   * Set all entries that are NULL to __math_dispatch_error.
   */

  for (f = 0; f < func_size; f++) {
    for (s = 0; s < sv_size; s++) {
      frp_e frp;
      for (frp = 0 ; frp < frp_size ; ++frp) {
        if (NULL == __mth_rt_vi_ptrs_new[f][s][frp]) {
          __mth_rt_vi_ptrs_new[f][s][frp] = __math_dispatch_error;
        }
      }
    }
  }

  /*
   * Now see if MTH_I_STATS is nonzero.
   * If so, enable statistics.
   */

  if (NULL != (ptenv = getenv("MTH_I_STATS"))) {
    char *pend;
    __mth_i_stats = strtoul(ptenv, &pend, 0);
    if (*ptenv != '\0' && *pend == '\0') {
      fprintf(stderr, "MTH_I_STATS=%" PRIu32 "\n", __mth_i_stats);
    } else {
      fprintf(stderr, "MTH_I_STATS=%s not a valid number - disabled\n", ptenv);
      __mth_i_stats = 0;
    }
  }

  if (__mth_i_stats != 0) {
    if (__mth_i_stats > stats_all) {
      fprintf(stderr, "MTH_I_STATS=%#x > %#x, defaulting to summary(%#x)\n",
        __mth_i_stats, stats_all, stats_summary);
      __mth_i_stats = stats_summary;
    }
#if defined(DISPATCH_IS_STATIC)
    fputs("MTH_I_STATS is enabled, but running with static "\
          "initialization\nMust call __math_epilog_ at program "\
          "termination to generate report\n", stderr);
#endif

    /*
     * To enable statistics:
     *
     * 1) The "default" dispatch table (__mth_rt_vi_ptrs_new) which has been
     *    previously populated is copied to __mth_rt_vi_ptrs_stat.
     *
     * 2) Copy the "statistics" dispatch table (__mth_rt_vi_ptrs_statdefs)
     *    which is normally not used over __mth_rt_vi_ptrs.
     *
     * So now the call flow becomes (use __fs_acos_1 as an example):
     *
     * A) entry in to __fs_acos_1:
     *    Get dispatch entry from __mth_rt_vi_ptrs[func_acos][sv_ss][frp_f]
     *    which points to __fs_acos_1_prof
     *    That is because all entries in __mth_rt_vi_ptrs now point to the
     *    profiled version of __fs_acos_1 (step 2 above)
     *    Jump to __fs_acos_1_prof
     *
     * B) entry in to __fs_acos_1_prof:
     *    Increment statistics for this call
     *    Get dispatch entry from __mth_rt_vi_ptrs_prof[func_acos][sv_ss][frp_f]
     *    which points to whatever __fs_acos_1 would have normally been defined
     *    to call.
     *    Jump to actual acos() entry point.
     */

    if (sizeof __mth_rt_vi_ptrs != sizeof __mth_rt_vi_ptrs_stat) {
      __pgmath_abort(1, "MTH_I_STATS: __mth_rt_vi_ptrs table size mismatch");
    }
    memset(__mth_rt_stats, 0, sizeof __mth_rt_stats);
    memcpy(__mth_rt_vi_ptrs_stat, __mth_rt_vi_ptrs_new, sizeof __mth_rt_vi_ptrs);
    memcpy(__mth_rt_vi_ptrs, __mth_rt_vi_ptrs_statdefs, sizeof __mth_rt_vi_ptrs_statdefs);
  } else {
    /*
     * Statistics disabled.
     */
    memcpy(__mth_rt_vi_ptrs, __mth_rt_vi_ptrs_new, sizeof __mth_rt_vi_ptrs);
  }


  if (__mth_i_debug & 0x1) {
    dump_mth_rt_vi_ptrs();
  }
}

/*
 * __math_dispatch_init() - initialize math dispatch tables when using
 * static libraries.
 *
 * __math_dispatch_init() should only be called from the
 * __[frp][sdcz]_<NAME>_<VL>_init() routines.
 *
 * We use environment variable MTH_I_DEBUG=0x100 to test that multiple
 * threads concurrently calling __math_dispatch_init get properly ordered.
 * That is only the first thread calls __math_dispatch() and the other
 * threads wait until dispatch setup has completed. A trivial OpenMP
 * program can be constructed to test parallel initialization.
 */

void
__math_dispatch_init()
{
  if (__sync_bool_compare_and_swap(&__math_dispatch_in_prog, false, true)) {
    if (__mth_i_debug == 0x100) {
      fputs("calling __math_dispatch()\n", stderr);
#if defined(TARGET_WIN)
      SLEEP(1);
#else
      struct timespec tsp = { 0, 250000000 };
      (void) nanosleep(&tsp, NULL);
#endif
    }
    __math_dispatch();
    __math_dispatch_is_init = true;
    __sync_synchronize();
  } else {
    if (__mth_i_debug == 0x100) {
      fputs("waiting for __math_dispatch\n", stderr);
    }
    while (false == __math_dispatch_is_init) {
#if     defined(TARGET_X8664)
      __asm__("pause");
#elif   defined(TARGET_LINUX_POWER) || defined(TARGET_ARM64)
      __asm__("yield");     // or   27,27,27
#else
      sched_yield();
#endif
    }
  }
}

void
__math_dispatch_error(void)
{
  static bool in_progress = false;

  /*
   * Simplistic way of only letting one thread through __math_dispatch_error().
   * Removes the need for complicated pthread / OpenMP / ... synchronization.
   *
   * Note: pgcc does not (yet?) have support for GNU/Intel atomic operations.
   *       So, calling statistics is not thread safe on OSX86-64 through
   *       __math_dispatch_error().
   */

#if !defined(__PGIC__)
  if ( false == __sync_bool_compare_and_swap(&in_progress, false, true)) {
#if !defined(TARGET_WIN)
    struct timespec tsp = { 0, 250000000 };
#endif
    while (true) {
#if defined(TARGET_WIN)
      SLEEP(1);     // The first thread will eventually abort the program
#else
      (void) nanosleep(&tsp, NULL); // The first thread will
				    // eventually abort the program
      tsp.tv_sec = 0;
      tsp.tv_nsec = 250000000;
#endif
    }
  }
#endif

  if (__mth_i_stats != 0) {
    __mth_i_stats |= stats_disp_err;     // Help user find incorrect call
    __math_epilog_do_stats();
  }
  fputs("Error during math dispatch processing...\n", stderr);
  fflush(stderr);
  __pgmath_abort(1, "Math dispatch table is either misconfigured or corrupted.");
}

#ifdef UNIT_TEST
static int
cmp_arch(const void *a, const void *b)
{
  mth_intrins_defs_t *pa = (__typeof__(pa))a;
  mth_intrins_defs_t *pb = (__typeof__(pa))b;

  return pa->arch < pb->arch ? -1 : pa->arch == pb->arch ? 0 : 1;
}

static int
cmp_func(const void *a, const void *b)
{
  mth_intrins_defs_t *pa = (__typeof__(pa))a;
  mth_intrins_defs_t *pb = (__typeof__(pa))b;

  return pa->func < pb->func ? -1 : pa->func == pb->func ? 0 : 1;
}

static int
cmp_sv(const void *a, const void *b)
{
  mth_intrins_defs_t *pa = (__typeof__(pa))a;
  mth_intrins_defs_t *pb = (__typeof__(pa))b;

  return pa->sv < pb->sv ? -1 : pa->sv == pb->sv ? 0 : 1;
}

int
main()
{
  int i, ip1;
  int teststatus = 0; // Not boolean - shell return status

  printf("main() - unit test invoked\n");
  qsort(mth_intrins_defs, sizeof(mth_intrins_defs) / sizeof(*mth_intrins_defs),
        sizeof(*mth_intrins_defs), cmp_sv);
  qsort(mth_intrins_defs, sizeof(mth_intrins_defs) / sizeof(*mth_intrins_defs),
        sizeof(*mth_intrins_defs), cmp_func);
  qsort(mth_intrins_defs, sizeof(mth_intrins_defs) / sizeof(*mth_intrins_defs),
        sizeof(*mth_intrins_defs), cmp_arch);
  for (i = 0, ip1 = 1;
       ip1 < sizeof(mth_intrins_defs) / sizeof(*mth_intrins_defs); i++, ip1++) {
    if (mth_intrins_defs[i].sv == mth_intrins_defs[ip1].sv) {
      printf("Appears to have duplicate entries index=%d\n", i);
      printf("func=%-5s type=%-5s arch=%-5s\n", cfunc[mth_intrins_defs[i].func],
             csv[mth_intrins_defs[i].sv], carch[mth_intrins_defs[i].arch]);
      teststatus = 1;
    }
  }

  return teststatus;
}

#undef MTHTMPDEF
#define MTHTMPDEF(_f) extern void _f(){};
#include "tmp-mth_alldefs.h"
#endif // UNIT_TEST

static void
__pgmath_abort(int ierr, char *s)
{
  fprintf(stderr, "__pgmath_abort:%s", s);
  exit(ierr);
}
