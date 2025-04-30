/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef SYMTAB_H_
#define SYMTAB_H_

/**
 *  \file
 *  \brief symtab.h - symbol tab definitions for Fortran
 */

#include <universal.h>

/* clang-format off */
.OC

.ST
/* the following macro depends on stype ordering */
#define ST_ISVAR(s) ( ((s) >= ST_VAR && (s) <= ST_UNION) )
#define IS_PROC(st) ( (st) >= ST_ENTRY && (st) <= ST_PD )
.TY

/*
 * DT_MAX before adding DT_DEFERNCHAR & DT_DEFERCHAR
 */
#define DT_MAX_43 43

/*
 * Target-dependent default integer, real, complex, and logical data types.
 * Double and double complex as well, since they may be quad.
 * These values are initialized in sym_init_first(); they need to be initialized
 * for the symtab utility.  The target-dependent values are assigned in
 * sym_init(); sym_init() has access to the flg structure.
 */

#define DT_INT stb.dt_int
#define DT_REAL stb.dt_real
#define DT_CMPLX stb.dt_cmplx
#define DT_LOG stb.dt_log
#define DT_DBLE stb.dt_dble
#define DT_DCMPLX stb.dt_dcmplx
#define DT_PTR stb.dt_ptr

#define DT_FLOAT DT_REAL4
#define TY_FLOAT TY_REAL
#define DT_CPTR DT_ADDR


#define DTY(d) (stb.dt.stg_base[d])

/* for fast DT checking -- define table indexed by TY_ */
extern short dttypes[TY_MAX+1];
.TA

#define DDTG(dt) (DTY(dt) == TY_ARRAY ? DTY(dt+1) : dt)
#define DTYG(dt) (DTY(dt) == TY_ARRAY ? DTY(DTY(dt+1)) : DTY(dt))

#define DT_ISCHAR(dt)	(dttypes[DTY(dt)]&(_TY_CHAR|_TY_NCHAR))
#define DT_ISINT(dt)	(dttypes[DTY(dt)]&_TY_INT)
#define DT_ISREAL(dt)	(dttypes[DTY(dt)]&_TY_REAL)
#define DT_ISCMPLX(dt)	(dttypes[DTY(dt)]&_TY_CMPLX)
#define DT_ISNUMERIC(dt) (dttypes[DTY(dt)]&(_TY_INT|_TY_REAL|_TY_CMPLX))
#define DT_ISBASIC(dt)	(dttypes[DTY(dt)]&_TY_BASIC)
#define DT_ISUNSIGNED(dt) (dttypes[DTY(dt)]&_TY_UNSIGNED)
#define DT_ISSCALAR(dt)	(dttypes[DTY(dt)]&_TY_SCALAR)
#define DT_ISVEC(dt)	(dttypes[DTY(dt)]&_TY_VEC)
#define DT_ISLOG(dt)	(dttypes[DTY(dt)]&_TY_LOG)
#define DT_ISWORD(dt)	(dttypes[DTY(dt)]&_TY_WORD)
#define DT_ISDWORD(dt)	(dttypes[DTY(dt)]&_TY_DWORD)

#define DT_ISINT_ARR(dt)     (DTY(dt)==TY_ARRAY && DT_ISINT(DTY(dt+1)))
#define DT_ISREAL_ARR(dt)    (DTY(dt)==TY_ARRAY && DT_ISREAL(DTY(dt+1)))
#define DT_ISCMPLX_ARR(dt)   (DTY(dt)==TY_ARRAY && DT_ISCMPLX(DTY(dt+1)))
#define DT_ISNUMERIC_ARR(dt) (DTY(dt)==TY_ARRAY && DT_ISNUMERIC(DTY(dt+1)))
#define DT_ISLOG_ARR(dt)     (DTY(dt)==TY_ARRAY && DT_ISLOG(DTY(dt+1)))

#define TY_ISCHAR(t)	(dttypes[t]&(_TY_CHAR|_TY_NCHAR))
#define TY_ISINT(t)	(dttypes[t]&_TY_INT)
#define TY_ISREAL(t)	(dttypes[t]&_TY_REAL)
#define TY_ISCMPLX(t)	(dttypes[t]&_TY_CMPLX)
#define TY_ISNUMERIC(t)	(dttypes[t]&(_TY_INT|_TY_REAL|_TY_CMPLX))
#define TY_ISBASIC(t)	(dttypes[t]&_TY_BASIC)
#define TY_ISUNSIGNED(t) (dttypes[t]&_TY_UNSIGNED)
#define TY_ISSCALAR(t)	(dttypes[t]&_TY_SCALAR)
#define TY_ISLOG(t)	(dttypes[t]&_TY_LOG)
#define TY_ISVEC(t)	(dttypes[t]&_TY_VEC)
#define TY_ISWORD(t)	(dttypes[t]&_TY_WORD)
#define TY_ISDWORD(t)	(dttypes[t]&_TY_DWORD)

#define IS_CHAR_TYPE(t) (TY_ISCHAR(t))

#define ALIGN(addr, a) ((addr + a) & ~(a))
#define ALIGN_AUTO(addr, a) ((addr) & ~(a))

.Sc

#define SC_AUTO SC_LOCAL
#define SC_ISCMBLK(p)  (p == SC_CMBLK)
#define CUDA_HOST             0x01
#define CUDA_DEVICE           0x02
#define CUDA_GLOBAL           0x04
#define CUDA_BUILTIN          0x08
#define CUDA_GRID             0x10

#define INTENT_IN 0x1
#define INTENT_OUT 0x2
#define INTENT_INOUT 0x3
#define INTENT_DFLT 0x0

#define IGNORE_T 0x1
#define IGNORE_K 0x2
#define IGNORE_R 0x4
#define IGNORE_D 0x8
/******          0x10 MARKER indicating IGNORE_TKR_ALL ******/
#define IGNORE_M 0x20
#define IGNORE_C 0x40
/* IGNORE_TKR directive without any specifiers, except for _C */
#define IGNORE_TKR_ALL 0x3f
#define IGNORE_TKR_ALL0 0x1f 	/* old value of IGNORE_TKR_ALL */

#define DLL_NONE   0x0
#define DLL_EXPORT 0x1
#define DLL_IMPORT 0x2

#define PRESCRIPTIVE  0
#define DESCRIPTIVE   1
#define TRANSCRIPTIVE 2

.Ik

#ifdef __cplusplus
inline SPTR check_SPTR(SPTR v) { return v; }
inline DTYPE check_DTYPE(DTYPE v) { return v; }
inline INT check_INT(INT v) { return v; }
#else
#define check_SPTR(v) (v)
#define check_DTYPE(v) (v)
#define check_INT(v) (v)
#endif
 
.SE

/* Test for type-bound procedure */
#define IS_TBP(func)  (VTOFFG(func) && TBPLNKG(func))

/* overloaded macros accessing shared fields */

#define FORALLNDXG(s)    DOVARG(s)
#define FORALLNDXP(s,v)  DOVARP(s,v)
#define ACONOFFG(s)   (( stb.stg_base)[s].w14)
#define ACONOFFP(s,v) (( stb.stg_base)[s].w14 = (v))
#define INTENTG(s)    b3G(s)
#define INTENTP(s,v)  b3P(s,v)
#define DLLG(s)       b3G(s)
#define DLLP(s,v)     b3P(s,v)
#define PDALN_EXPLICIT_0 0xf
#define PDALNG(s)     (b4G(s) == PDALN_EXPLICIT_0 ? 0 : b4G(s))
#define PDALNP(s,v)   b4P(s, (v) == 0 ? PDALN_EXPLICIT_0 : (v))
#define PDALN_IS_DEFAULT(s) (b4G(s) == 0)
#define CUDAG(s)      b4G(s)
#define CUDAP(s,v)    b4P(s,v)
#define NEWDSCG(s)   (( stb.stg_base)[s].w10)
#define NEWDSCP(s,v) (( stb.stg_base)[s].w10 = (v))
#define ARGINFOG(s)   (( stb.stg_base)[s].w16)
#define ARGINFOP(s,v) (( stb.stg_base)[s].w16 = (v))
#define XREFLKG(s)   (( stb.stg_base)[s].w16)
#define XREFLKP(s,v) (( stb.stg_base)[s].w16 = (v))

#define SYMNAME(p)        (stb.n_base + NMPTRG(p))
#define SYMNAMEG(p, buff, len)    len = NMLENG(p); strncpy(buff,SYMNAME(p),len)
#define LOCAL_SYMNAME(p) local_sname(SYMNAME(p))
#define RFCNTI(s) (++RFCNTG(s))
#define RFCNTD(s) (--RFCNTG(s))
#define KWDARGSTR(p) intrinsic_kwd[KWDARGG(p)]
#define NOSYM 1

typedef enum{
    ETLS_PROCESS,
    ETLS_TASK,
    ETLS_THREAD,
    ETLS_OMP
} etls_levels;
#define IS_TLS(s) (TLSG(s) || ETLSG(s))
#define IS_THREAD_TLS(s) (THREADG(s) &&   IS_TLS(s))
#define IS_THREAD_TP(s)  (THREADG(s) && (!IS_TLS(s)))

#define CMPLXFUNC_C XBIT(49, 0x40000000)

/*****  Array Descriptor  *****/

typedef struct {
    int    numdim;
    int    zbase;
    char   pxx[2];	/* available flags */
    char   assumrank;
    char   assumshp;
    char   defer;
    char   adjarr;
    char   assumsz;
    char   nobounds;
    struct {
        int mlpyr;
        int lwbd;
        int upbd;
	int lwast;
	int upast;
	int extntast;
    } b[1];
} ADSC;

#define AD_DPTR(dtype) ((ADSC *)(aux.arrdsc_base+DTY((dtype)+2)))
#define AD_PTR(sptr) ((ADSC *) (aux.arrdsc_base + DTY(DTYPEG(sptr)+2)))
#define AD_NUMDIM(p)  ((p)->numdim)
#define AD_DEFER(p) ((p)->defer)
#define AD_ASSUMRANK(p) ((p)->assumrank)
#define AD_ASSUMSHP(p) ((p)->assumshp)
#define AD_ADJARR(p) ((p)->adjarr)
#define AD_ASSUMSZ(p) ((p)->assumsz)
#define AD_NOBOUNDS(p) ((p)->nobounds)
#define AD_ZBASE(p)  ((p)->zbase)
#define AD_MLPYR(p, i) ((p)->b[i].mlpyr)
#define AD_LWBD(p, i)  ((p)->b[i].lwbd)
#define AD_UPBD(p, i)  ((p)->b[i].upbd)
#define AD_LWAST(p, i)  ((p)->b[i].lwast)
#define AD_UPAST(p, i)  ((p)->b[i].upast)
#define AD_EXTNTAST(p, i) ((p)->b[i].extntast) 
#define AD_NUMELM(p)  ((p)->b[AD_NUMDIM(p)].mlpyr)

/* Use the following macros instead of the AD_* macros when there is
 * the possibility of memory reallocation following assignment of an
 * array descriptor's address to a pointer. */
#define ADD_NUMDIM(dtyp)  AD_NUMDIM(AD_DPTR(dtyp))
#define ADD_DEFER(dtyp) AD_DEFER(AD_DPTR(dtyp))
#define ADD_ASSUMRANK(dtyp) AD_ASSUMRANK(AD_DPTR(dtyp))
#define ADD_ASSUMSHP(dtyp) AD_ASSUMSHP(AD_DPTR(dtyp))
#define ADD_ADJARR(dtyp) AD_ADJARR(AD_DPTR(dtyp))
#define ADD_ASSUMSZ(dtyp) AD_ASSUMSZ(AD_DPTR(dtyp))
#define ADD_NOBOUNDS(dtyp) AD_NOBOUNDS(AD_DPTR(dtyp))
#define ADD_ZBASE(dtyp)  AD_ZBASE(AD_DPTR(dtyp))
#define ADD_MLPYR(dtyp, i) AD_MLPYR(AD_DPTR(dtyp), i)
#define ADD_LWBD(dtyp, i)  AD_LWBD(AD_DPTR(dtyp), i)
#define ADD_UPBD(dtyp, i)  AD_UPBD(AD_DPTR(dtyp), i)
#define ADD_LWAST(dtyp, i)  AD_LWAST(AD_DPTR(dtyp), i)
#define ADD_UPAST(dtyp, i)  AD_UPAST(AD_DPTR(dtyp), i)
#define ADD_EXTNTAST(dtyp, i)  AD_EXTNTAST(AD_DPTR(dtyp), i) 
#define ADD_NUMELM(dtyp)  AD_NUMELM(AD_DPTR(dtyp))

typedef struct {
    ISZ_T   stack_addr;	/* available address on run-time stack  */
    int     ent_save;	/* sptr to cc array to hold saved ar's and excstat */
    short   first_dr;	/* first data reg used as global  */
    short   first_ar;	/* first address reg used as global  */
    short   first_sp;	/* first float reg used as global  */
    short   first_dp;	/* first double reg used as global  */
    int     auto_array;	/* static array used for auto vars, else 0 */
    int     ret_var;   	/* sym of return value if passed as arg */
    int     memarg_ptr;	/* sym where memarg ptr is saved upon entry */
    int     gr_area;	/* sym of where to save global regs */
    int     flags;	/* misc. target dependent flags */
    char   *arasgn;	/* local ar (base pointer) ARASGN records */
    char   *regset;	/* target dependent register set info */
    char   *argset;	/* target dependent register set info */
    int     launch_maxthread, launch_minctasm;
                        /* launch_bounds for CUDA Fortran. 0 means not set. */
} ENTRY;


/*****  Namelist Descriptor  *****/

typedef struct {
    int   sptr;
    int   next;
    int   lineno;
} NMLDSC;

#define NML_SPTR(i)   aux.nml_base[i].sptr
#define NML_NEXT(i)   aux.nml_base[i].next
#define NML_LINENO(i) aux.nml_base[i].lineno


/*****  Symbol List Item  *****/

typedef struct {
    int   sptr;
    int   next;
} SYMI;

#define SYMI_SPTR(i) aux.symi_base[i].sptr
#define SYMI_NEXT(i) aux.symi_base[i].next

/*
 * Define macro which converts character into index into implicit array.
 */
#define IMPL_INDEX(uc)  (islower(uc) ? uc - 'a' :      \
			   (isupper(uc) ? 26+(uc-'A') : \
			      (uc == '$'  ? 52 :              \
				 (uc == '_'  ?  53  : -1) )))

typedef struct {
    int    sptr;
    INT    conval;
} DVL;

#define DVL_SPTR(i)   aux.dvl_base[i].sptr
#define DVL_CONVAL(i) aux.dvl_base[i].conval

typedef struct {
   int    *dpdsc_base;
   int     dpdsc_size;
   int     dpdsc_avl;
   int    *arrdsc_base;
   int     arrdsc_size;
   int     arrdsc_avl;
   ENTRY  *entry_base;
   int     entry_size;
   int     entry_avail;
   ENTRY  *curr_entry;
   int     dt_iarray;
   int     dt_iarray_int;
   NMLDSC *nml_base;
   int     nml_size;
   int     nml_avl;
   DVL    *dvl_base;
   int     dvl_size;
   int     dvl_avl;
   int     list[ST_MAX+1];
   SYMI   *symi_base;
   int     symi_size;
   int     symi_avl;
   INT    *parsyms_base; /* Symbols in parallel regions */
   int     parsyms_size;
   int     parsyms_avl;
} AUX;
 
#include "symacc.h"

/*   symbol table data declarations:  */
 
extern AUX aux;

extern const char *intrinsic_kwd[];

/*  declarations required to access switch statement or computed goto lists: */

typedef struct {
    INT   val;
    INT   uval;		/* val : uval, for CASE constructs */
    SPTR  clabel;
    int   next;
} SWEL;

extern SWEL *switch_base;

/*   declare external functions from symtab.c */

void sym_init(void);
void init_implicit(void);
void implicit_int(int);
void save_implicit(LOGICAL);
void restore_implicit(void);
void hpf_library_stat(int *, int *, int);
SPTR getsym (const char *, int);
SPTR getsymbol (const char *);
SPTR getsymf(const char *, ...);
SPTR getcon(INT *, DTYPE);
SPTR get_acon(SPTR, ISZ_T);
SPTR get_acon3(SPTR, ISZ_T, DTYPE);
INT get_int_cval(int);
ISZ_T get_isz_cval(int);
INT sign_extend(INT, int);
int getstring(const char *, int);
int gethollerith(int, int);
void newimplicit(int, int, int);
void setimplicit(int);
const char *parmprint(int);
const char *getprint(int);
void symdentry(FILE *, int);
void symdmp(FILE *, LOGICAL);
void dmp_socs(int, FILE *);
int getccsym(int, int, SYMTYPE);
int getnewccsym(int, int, int);
int getnewccsymf(int stype, const char *, ...);
int getccsym_sc(int, int, int, int);
int getccssym(const char *, int, int);
int getccssym_sc(const char *, int, int, int);
int getcctmp(int, int, int, int);
int getcctmp_sc(int, int, int, int, int);
int insert_sym(int);
int insert_sym_first(int);
int getlab(void);
void pop_sym(int);
SPTR mkfunc(const char *);
SPTR mkfunc_cncall(const char *);
char *mkfunc_name(int, char *);
char *mkfunc_ir8name(int, char *, int);
char *ir8name(char *, int);
const char *mk_coercion_func_name(int);
int mk_coercion_func(int);
int mk_external_var(char *, int);
LOGICAL is_arg_in_entry(int, int);
int resolve_sym_aliases(int);
LOGICAL is_procedure_ptr(int);
void proc_arginfo(int, int *, int *, int *);
void copy_sym_flags(SPTR, SPTR);
void dup_sym(int, struct SYM *);
int insert_dup_sym(int);
int get_align_desc(int, int);
void dump_align(FILE*, int);
int copy_align_desc(int);
int copy_dist_desc(int);
int get_dist_desc(int);
void dump_dist(FILE*, int);
int get_shadow_desc(int);
void dump_shadow(FILE*, int);
char *mangle_name(const char *, const char *);
char *sym_strsave(const char *);
void save_uname(int, INT);
int add_symitem(int, int);
void change_predefineds(int, LOGICAL);
SPTR find_explicit_interface(SPTR s);
SPTR instantiate_interface(SPTR iface);
void convert_2dollar_signs_to_hyphen(char *name);

char *getsname(int);	/* defined in assem.c */
void sym_is_refd(SPTR);	/* defined in assem.c */

void iso_c_lib_stat(int *, int *, int);
int get_ieee_arith_intrin(const char *);
void symtab_standard(void);
void symtab_nostandard(void);
void newimplicitnone(void);
void reinit_sym(int);
void symtab_fini(void);
int get_len_of_deferchar_ast(int ast); /* symutl.c */
SPTR get_proc_ptr(SPTR sptr); /* symutl.c */

LOGICAL sym_in_sym_list(int sptr, int symi);
LOGICAL same_sym_list(int list1, int list2);
void push_sym(int sptr);
void init_implicit(void);
void implicit_int(int default_int);
bool cmp_interfaces(int sym1, int sym2, int flag);
void dmp_socs(int sptr, FILE *file);

#define ALIGNG(sptr)	0
#define DISTG(sptr)	0
#define RUNTIMEG(sptr)	0
#define INHERITG(sptr)	0

/**
 * \brief flag defintions for cmp_interfaces_strict()
 */
typedef enum CMP_INTERFACE_FLAGS {
  IGNORE_IFACE_NAMES = 0x0, /**< ignore the symbol names sym1 & sym2, but make
                                 sure arguments have same stypes and names. */
  CMP_IFACE_NAMES = 0x1, /**< make sure sym1 and sym2 have the same symbol 
                              name. */                          
  IGNORE_ARG_NAMES = 0x2, /**< ignore the argument names. */   
  RELAX_STYPE_CHK = 0x4, /**< relax stype check on arguments. */
  CMP_OPTARG = 0x8, /**< make sure sym1 and sym2 OPTARG fields are identical. */
  RELAX_INTENT_CHK = 0x10, /**< relax intent check on arguments. */
  RELAX_POINTER_CHK = 0x20, /**< relax pointer check on arguments. */

  RELAX_PURE_CHK_1 = 0x40, /**< relax pure check on argument #1 of
                                cmp_interfaces_strict() function */
  RELAX_PURE_CHK_2 = 0x80,  /**< relax pure check on argument #2 of
                                cmp_interfaces_strict() function */
  CMP_SUBMOD_IFACE = 0x100, /**< make sure submodule interface of a procedure 
                                 defined by a separate module subprogram's 
                                 definition matches the declaration */
  DEFER_IFACE_CHK = 0x200   /**< defer interface check for procedure dummy
                                 arguments. */
} cmp_interface_flags;

bool compatible_characteristics(int psptr, int psptr2,
                                cmp_interface_flags flag);
bool cmp_interfaces_strict(SPTR sym1, SPTR sym2, cmp_interface_flags flag);
bool is_used_by_submod(SPTR sym1, SPTR sym2);

#endif // SYMTAB_H_
