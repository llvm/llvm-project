/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef SYMTAB_H_
#define SYMTAB_H_

/** \file
 *  \brief symbol table for Fortran backend
 */

#include "global.h"
#include <stdarg.h>

/* clang-format off */
.OC

.ST
/* the following macro depends on stype ordering */
#define ST_ISVAR(s) ((s) >= ST_VAR && (s) <= ST_UNION)

.TY

#define DT_FLOAT DT_REAL
#define TY_FLOAT TY_REAL
#define DT_CPTR DT_ADDR

#define DTY(d) (stb.dt.stg_base[d])

/* for fast DT checking -- define table indexed by TY_ */
extern short dttypes[TY_MAX+1];
.TA

#define DDTG(dt) (DTY(dt) == TY_ARRAY ? DTY(dt+1) : dt)
#define DTYG(dt) (DTY(dt) == TY_ARRAY ? DTY(DTY(dt+1)) : DTY(dt))

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
#define DT_ISVECT(dt)   (dttypes[DTY(dt)]&_TY_VECT)

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
#define TY_ISVECT(t)    (dttypes[t]&_TY_VECT)

#define TY_VECT_MAXLEN 16

#define ALIGN(addr, a) ((addr + a) & ~(a))
#define ALIGN_AUTO(addr, a) ((addr) & ~(a))

.Sc

#define SC_AUTO SC_LOCAL
#define SC_ISCMBLK(p)  (p == SC_CMBLK)

.SE

/* redo & add a few macros when BIGOBJects are allowed.
 * The value of the CONVAL2 field is 'assumed' to be a 32-bit;
 * the offset of an address constant is 64-bit.
 */

#undef CONVAL2G
#undef CONVAL2P
#undef CMEMLG
#undef CMEMLP
#undef DEFLABG
#undef DEFLABP
#undef ENDLINEG
#undef ENDLINEP
#undef FUNCLINEG
#undef FUNCLINEP
#undef GSAMEG
#undef GSAMEP
#undef ILMG
#undef ILMP
#define CONVAL2G(s)   (INT)(( stb.stg_base)[s].w14)
#define CONVAL2P(s,v) (INT)(( stb.stg_base)[s].w14 = (v))
#define CMEMLG(s)   (INT)(( stb.stg_base)[s].w14)
#define CMEMLP(s,v) (INT)(( stb.stg_base)[s].w14 = (v))
#define DEFLABG(s)   (INT)(( stb.stg_base)[s].w14)
#define DEFLABP(s,v) (INT)(( stb.stg_base)[s].w14 = (v))
#define ENDLINEG(s)   (INT)(( stb.stg_base)[s].w14)
#define ENDLINEP(s,v) (INT)(( stb.stg_base)[s].w14 = (v))
#define FUNCLINEG(s)   (INT)(( stb.stg_base)[s].w14)
#define FUNCLINEP(s,v) (INT)(( stb.stg_base)[s].w14 = (v))
#define GSAMEG(s)   (INT)(( stb.stg_base)[s].w14)
#define GSAMEP(s,v) (INT)(( stb.stg_base)[s].w14 = (v))
#define ILMG(s)   (INT)(( stb.stg_base)[s].w14)
#define ILMP(s,v) (INT)(( stb.stg_base)[s].w14 = (v))

#undef ORIGDUMMYG
#undef ORIGDUMMYP
#define ORIGDUMMYG(s)   (INT)(( stb.stg_base)[s].w32)
#define ORIGDUMMYP(s,v) (INT)(( stb.stg_base)[s].w32 = (v))

#undef GREALG
#undef GREALP
#undef SFDSCG
#undef SFDSCP
#define GREALG(s)   (INT)(( stb.stg_base)[s].w10)
#define GREALP(s,v) (INT)(( stb.stg_base)[s].w10 = (v))
#define SFDSCG(s)   (INT)(( stb.stg_base)[s].w10)
#define SFDSCP(s,v) (INT)(( stb.stg_base)[s].w10 = (v))

/* overloaded macros accessing shared fields */

#define ACONOFFG(s)   (( stb.stg_base)[s].w14)
#define ACONOFFP(s,v) (( stb.stg_base)[s].w14 = (v))
#define PARAMVALG(s)   (( stb.stg_base)[s].w15)
#define PARAMVALP(s,v) (( stb.stg_base)[s].w15 = (v))
#define DLLG(s)       b3G(s)
#define DLLP(s,v)     b3P(s,v)
#define PDALN_EXPLICIT_0 0xf
#define PDALNG(s)     ((b4G(s)&0x0f) == PDALN_EXPLICIT_0 ? 0 : (b4G(s)&0x0f))
#define PDALNP(s,v)   b4P(s, (b4G(s)&0xf0) | ((v) == 0 ? PDALN_EXPLICIT_0 : (v)))
#define PDALN_IS_DEFAULT(s) ((b4G(s)&0x0f) == 0)
#ifdef PGF90
#define CUDAG(s)      b4G(s)
#define CUDAP(s,v)    b4P(s,v)
#define CUDA_HOST		0x01
#define CUDA_DEVICE		0x02
#define CUDA_GLOBAL		0x04
#define CUDA_BUILTIN		0x08
#define CUDA_GRID		0x10
#define CUDA_CONSTRUCTOR	0x20
#define CUDA_STUB		0x40
 /* b4G and b4P can only be up to 0xFF */
#endif

#define SYMNAME(p)        (stb.n_base + NMPTRG(p))
#define SYMNAMEG(p, buff, len)    len = NMLENG(p); strncpy(buff,SYMNAME(p),len)
#define LOCAL_SYMNAME(p) local_sname(SYMNAME(p))
#define RFCNTI(s) (++RFCNTG(s))
#define RFCNTD(s) (--RFCNTG(s))
#define RETADJG(s)      (( stb.stg_base)[s].w10)
#define RETADJP(s,v)    (( stb.stg_base)[s].w10 = (v))
#define XREFLKG(s)      (( stb.stg_base)[s].w16)
#define XREFLKP(s,v)    (( stb.stg_base)[s].w16 = (v))
#define NOSYM ((SPTR)1)

typedef enum etls_levels {
    ETLS_PROCESS,
    ETLS_TASK,
    ETLS_THREAD,
    ETLS_OMP,
    /* Insert HLS here ?*/
    ETLS_NUM_LEVELS
} etls_levels;

#define IS_TLS(s) (TLSG(s) || ETLSG(s))
#define IS_THREAD_TLS(s) (THREADG(s) &&   IS_TLS(s))
#define IS_THREAD_TP(s)  (THREADG(s) && (!IS_TLS(s)))
#define IS_TLS_WRAPPER(sptr) (0)
#define IS_TLS_GETTER(sptr) IS_TLS(sptr)

#define CMPLXFUNC_C XBIT(49, 0x40000000)

#define DLL_NONE   0x0
#define DLL_EXPORT 0x1
#define DLL_IMPORT 0x2

typedef struct ADSC {
  int   numdim;
  int   scheck;
  int   zbase;
  SPTR  sdsc;
  ILM_T *ilmp;
  struct {
    SPTR mlpyr;
    SPTR lwbd;
    SPTR upbd;
  } b[1];
} ADSC;

#define AD_DPTR(dtype) ((ADSC *)(aux.arrdsc_base+DTY((dtype)+2)))
#define AD_PTR(sptr) ((ADSC *) (aux.arrdsc_base + DTY(DTYPEG(sptr)+2)))
#define AD_NUMDIM(p)  ((p)->numdim)
#define AD_SCHECK(p) ((p)->scheck)
#define AD_ZBASE(p)  ((p)->zbase)
#define AD_SDSC(p)   ((p)->sdsc)
#define AD_ILMP(p)   ((p)->ilmp)
#define AD_MLPYR(p, i) ((p)->b[i].mlpyr)
#define AD_LWBD(p, i)  ((p)->b[i].lwbd)
#define AD_UPBD(p, i)  ((p)->b[i].upbd)
#define AD_NUMELM(p)  ((p)->b[AD_NUMDIM(p)].mlpyr)

typedef struct ENTRY {
  ISZ_T  stack_addr; /* available address on run-time stack  */
  SPTR   ent_save;	/* sptr:
                         * o  n10 - to cc array to hold saved ar's and
                         *    excstat
                         * o  x86 - to cc scalar if multiple entries.
                         */
  short  first_dr;	/* first data reg used as global  */
  short           first_ar;	/* first address reg used as global  */
  short           first_sp;	/* first float reg used as global  */
  short           first_dp;	/* first double reg used as global  */
  int             auto_array; /* static array used for auto vars, else 0 */
  int             ret_var;   	/* sym of return value if passed as arg */
  int             memarg_ptr; /* sym where memarg ptr is saved upon entry */
  int             gr_area;    /* sym of where to save global regs */
  INT             flags;	/* misc. target dependent flags */
  char           *arasgn;	/* local ar (base pointer) ARASGN records */
  char           *regset;	/* target dependent register set info */
  char           *argset;	/* target dependent register set info */
  SPTR            display;    /* sptr to an internal procedure's display
                               * (i.e., the host procedure's stack frame).
                               */
  SPTR         uplevel;    /* sptr to an outlined function contains
                            * addresses of uplevel variables /
                            */
  int             cgr;	/* index into the simplfied call graph info */
  int     launch_maxthread, launch_minctasm;
                        /* launch_bounds for CUDA Fortran. 0 means not set. */
} ENTRY;

typedef struct NMLDSC {
    int   sptr;
    int   next;
    int   lineno;
} NMLDSC;

#define NML_SPTR(i)   aux.nml_base[i].sptr
#define NML_NEXT(i)   aux.nml_base[i].next
#define NML_LINENO(i) aux.nml_base[i].lineno

/*****  Symbol List Item  *****/

typedef struct SYMI {
    int   sptr;
    int   next;
} SYMI;

#define SYMI_SPTR(i) aux.symi_base[i].sptr
#define SYMI_NEXT(i) aux.symi_base[i].next


typedef struct DVL {
    int    sptr;
    INT    conval;
} DVL;

#define DVL_SPTR(i)   aux.dvl_base[i].sptr
#define DVL_CONVAL(i) aux.dvl_base[i].conval

typedef struct AUX {
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
   int     strdesc;
   NMLDSC *nml_base;
   int     nml_size;
   int     nml_avl;
   DVL    *dvl_base;
   int     dvl_size;
   int     dvl_avl;
   SYMI   *symi_base;
   int     symi_size;
   int     symi_avl;
   INT    *vcon_base;
   int     vcon_size;
   int     vcon_avl;
   int     parregs;      /* Number of parallel regions  */
   INT    *parsyms_base; /* Symbols in parallel regions */
   int     parsyms_size;
   int     parsyms_avl;
   int     vtypes[TY_MAX+1][TY_VECT_MAXLEN];
} AUX;

#define VCON_CONVAL(i) aux.vcon_base[i]

#include "symacc.h"

/*   symbol table data declarations:  */

extern AUX aux;

/* pointer-sized integer */

#define __POINT_T DT_INT8

/*  declarations required to access switch statement or computed goto lists: */

typedef struct SWEL {
    INT  val;
    SPTR clabel;
    int  next;
} SWEL;

extern SWEL *switch_base;

/**
   \brief ...
 */
ISZ_T get_isz_cval(int con);

/**
   \brief ...
 */
char *getprint(int sptr);

/**
   \brief ...
 */
const char *parmprint(int sptr);

/**
   \brief Add a new symbol with same name as an existing symbol
   \param oldsptr symbol to duplicate
 */
SPTR adddupsym(SPTR oldsptr);

/**
   \brief Add a new symbol with given name
   \param name  the symbol's name
 */
SPTR addnewsym(const char *name);

/**
   \brief ...
 */
int add_symitem(int sptr, int nxt);

/**
   \brief ...
 */
int dbg_symdentry(int sptr);

/**
   \brief Create (or possibly reuse) a compiler created symbol whose name is of
   the form . <pfx> dddd where dddd is the decimal representation of n.
 */
SPTR getccssym(const char *pfx, int n, SYMTYPE stype);

/**
   \brief Similar to getccssym, but storage class is an argument. Calls
   getccssym if the storage class is not private; if private, a 'p' is appended
   to the name.
 */
SPTR getccssym_sc(const char *pfx, int n, SYMTYPE stype, SC_KIND sc);

/**
   \brief ...
 */
SPTR getccsym_copy(SPTR oldsptr);

/**
   \brief create (or possibly reuse) a compiler created symbol whose name is of
   the form . <letter> dddd where dddd is the decimal representation of n.
 */
SPTR getccsym(char letter, int n, SYMTYPE stype);

/**
   \brief Similar to getccsym, but storage class is an argument. Calls
   getccsym if the storage class is not private; if private, a 'p' is
   appended to the name.
 */
SPTR getccsym_sc(char letter, int n, SYMTYPE stype, SC_KIND sc);

/**
   \brief Create (or possibly reuse) a compiler created temporary where the
   caller constructs the name and passes the storage class as an argument.
 */
SPTR getcctemp_sc(const char *name, SYMTYPE stype, SC_KIND sc);

/**
   \brief ...
 */
int get_entry_item(void);

/**
   \brief ...
 */
SPTR getlab(void);

/**
   \brief Create (never reuse) a compiler created symbol whose name is of the
   form . <letter> dddd where dddd is the decimal representation of n.
 */
SPTR getnewccsym(char letter, int n, SYMTYPE stype);

/**
   \brief ...
 */
SPTR get_semaphore(void);

/**
   \brief Enter character constant into symbol table
   \param value is the character string value
   \param length is the length of character string
   \return a pointer to the character constant in the symbol table.

   If the constant was already in the table, returns a pointer to the existing
   entry instead.
 */
SPTR getstring(const char *value, int length);

/**
   \brief Similar to getstring except the character string is null terminated
 */
SPTR getntstring(const char *value);

SPTR getstringaddr(SPTR sptr);

/**
   \brief ...
 */
int get_vcon0(DTYPE dtype);

/**
   \brief ...
 */
int get_vcon1(DTYPE dtype);

/**
   \brief ...
 */
SPTR get_vcon(INT *value, DTYPE dtype);

/**
   \brief get a vector constant of a zero which suits the element type
 */
int get_vconm0(DTYPE dtype);

/**
   \brief ...
 */
SPTR get_vcon_scalar(INT sclr, DTYPE dtype);

/**
   \brief ...
 */
SPTR insert_sym_first(SPTR first);

/**
   \brief ...
 */
SPTR insert_sym(SPTR first);

/**
   \brief ...
 */
SPTR mk_prototype(const char *name, const char *attr, DTYPE resdt, int nargs,
                  ...);

/**
   \brief ...
 */
SPTR mk_prototype_llvm(const char *name, const char *attr, DTYPE resdt,
                       int nargs, ...);

/**
   \brief ...
 */
INT sign_extend(INT val, int width);

/**
   \brief ...
 */
int tr_conval2g(char *fn, int ln, int s);

/**
   \brief ...
 */
int tr_conval2p(char *fn, int ln, int s, int v);

/**
   \brief ...
 */
SPTR get_acon3(SPTR sym, ISZ_T off, DTYPE dtype);

/**
   \brief ...
 */
SPTR get_acon(SPTR sym, ISZ_T off);

/**
   \brief ...
 */
SPTR getcon(INT *value, DTYPE dtype);

/**
   \brief ...
 */
SPTR getsymbol(const char *name);

/**
   \brief ...
 */
SPTR getsym(const char *name, int olength);

/**
   \brief ...
 */
SPTR mkfunc(const char *nmptr);

/**
   \brief ...
 */
void dmp_socs(int sptr, FILE *file);

/**
   \brief ...
 */
void implicit_int(DTYPE default_int);

/**
   \brief Change settings for implicit variable types, character lengths
   \param firstc   characters delimiting range
   \param lastc    characters delimiting range
   \param dtype    new value assigned to range
 */
void newimplicit(int firstc, int lastc, DTYPE dtype);

/**
   \brief ...
 */
void pop_scope(void);

/**
   \brief ...
 */
void pop_sym(int sptr);

/**
   \brief ...
 */
void reapply_implicit(void);

/**
   \brief ...
 */
void setimplicit(int sptr);

/**
   \brief ...
 */
void symdentry(FILE *file, int sptr);

/**
   \brief ...
 */
void symdmp(FILE *dfil, bool full);

/**
   \brief ...
 */
void sym_init(void);

#ifdef __cplusplus
// FIXME - these are hacks to allow addition on DTYPEs
inline DTYPE operator+=(DTYPE d, int c)
{
  return static_cast<DTYPE>(static_cast<int>(d) + c);
}

inline int operator+(DTYPE d, int c)
{
  return static_cast<int>(d) + c;
}
#endif

#endif
