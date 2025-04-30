/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file semstk.h
 * Shared by Parser and Semantic Analyzer modules.
 * This file contains semantic stack definitions and prototypes of functions
 * that operate on the semantic stack.
 */

/*
 * These two macros refer to the translation associated with the
 * Left Hand Side non-terminal and the Right Hand Side elements
 * numbers 1 to N
 */
#define LHS top
#define RHS(i) (top + ((i)-1))

/* stack size */
#define SST_SIZE 200

/* INT members within the union portion of the stack entry need to be
 * padded for systems, such as the decalpha, where ints are 32-bits and
 * pointers are 64-bits.  We cannot have pointer members in the stack entry
 * overlapping two INT members.
 */
#define SST_INT(m) INT m

typedef struct sst {
  short id;           /**< type of this stack entry */
  unsigned char flag; /**< general flag */
  unsigned f1 : 1;    /**< plain expr flag - 0 => no parens */
  unsigned f2 : 1;    /**< id is an alias */
  int ast; /**< the AST for this stack entry */ 
  int mnoff; /**< derived type flag & information */
  int sr;    /**< save & restore word */
  int lineno; /**< line number associated with this stack entry */
  int col;    /**< column number associated with this stack entry */

  union { /**< value of this stack entry */
    struct {/**< general purpose word value */
      SST_INT(w1);
      SST_INT(w2);
      SST_INT(w3);
      SST_INT(w4);
      SST_INT(w5);
    } wval;
    struct {         /**< constructor value */
      SST_INT(dum1); /**< needs wval.w1 */
      SST_INT(dum2); /**< needs wval.w2 */
      SST_INT(dum3); /**< needs wval.w3 */
      SST_INT(dum4); /**< needs wval.w4 */
      ACL *acl;
    } cnval;
    struct {         /**< equivalence item */
      SST_INT(dum1); /**< needs wval.w1 (SYM)*/
      SST_INT(substring);
      SST_INT(offset);
      SST_INT(subscript);
    } eqvval;
    struct {/**< item list value */
      ITEM *beg;
      ITEM *end;
      SST_INT(count);
    } ilval;
    struct {/**< variable name list for initializers */
      VAR *beg;
      VAR *end;
      SST_INT(count);
    } vlval;
    struct {/**< constant list for initializers */
      ACL *beg;
      ACL *end;
      SST_INT(count);
    } clval;
    struct {      /**< derived type value */
      ITEM *dum1; /**< needs ilval.beg */
      ITEM *dum2; /**< needs ilval.end */
      ITEM *beg;
      ITEM *end;
    } dtval;
    struct {/**< vector slice triplet notation */
      struct sst *next;
      struct sst *e1;
      struct sst *e2;
      struct sst *e3;
    } tlval;
  } value;
} SST;

extern SST *sst;

/* ident put/get macros -- for all types */
#define SST_IDG(p) ((p)->id)
#define SST_IDP(p, v) ((p)->id = ((int)(v)))

#define SST_FLAGG(p) ((p)->flag)
#define SST_FLAGP(p, v) ((p)->flag = ((int)(v)))

#define SST_PARENG(p) ((p)->f1)
#define SST_PARENP(p, v) ((p)->f1 = ((int)(v)))

#define SST_ALIASG(p) ((p)->f2)
#define SST_ALIASP(p, v) ((p)->f2 = (v))

/* put/get macros for ast */
#define SST_ASTG(p) ((p)->ast)
#define SST_ASTP(p, v) ((p)->ast = (v))

#define SST_MNOFFG(p) ((p)->mnoff)
#define SST_MNOFFP(p, v) ((p)->mnoff = (v))

#define SST_DIMFLAGG(p) ((p)->mnoff)
#define SST_DIMFLAGP(p, v) ((p)->mnoff = (v))

#define SST_TMPG(p) ((p)->sr)
#define SST_TMPP(p, v) ((p)->sr = (v))

#define SST_LINENOG(p) ((p)->lineno)
#define SST_LINENOP(p, v) ((p)->lineno = (v))

#define SST_COLUMNG(p) ((p)->col)
#define SST_COLUMNP(p, v) ((p)->col = (v))


/* put/get macros for expressions */
#define SST_OPTYPEG(p) ((p)->value.wval.w1)
#define SST_SYMG(p) ((p)->value.wval.w1)
#define SST_CVALG(p) ((p)->value.wval.w1)
#define SST_GDTYPEG(p) ((p)->value.wval.w2)
#define SST_DTYPEG(p) ((p)->value.wval.w2)
#define SST_GTYG(p) ((p)->value.wval.w3)
#define SST_LSYMG(p) ((p)->value.wval.w3)
#define SST_LENG(p) ((p)->value.wval.w3)
#define SST_ERRSYMG(p) ((p)->value.wval.w3)
#define SST_SHAPEG(p) ((p)->value.wval.w4)
#define SST_OPCG(p) ((p)->value.wval.w4)
#define SST_UNITG(p) ((p)->value.wval.w4)
#define SST_FIRSTG(p) ((p)->value.wval.w4)
#define SST_LASTG(p) ((p)->value.wval.w5)
#define SST_CVLENG(p) ((p)->value.wval.w5)
#define SST_CPLXPARTG ((p)->value.wval.w5)
#define SST_ACLG(p) ((p)->value.cnval.acl)
#define SST_SUBSCRIPTG(p) ((p)->value.eqvval.subscript)
#define SST_SUBSTRINGG(p) ((p)->value.eqvval.substring)
#define SST_OFFSETG(p) ((p)->value.eqvval.offset)
#define SST_NMLBEGG(p) ((p)->value.wval.w1)
#define SST_NMLENDG(p) ((p)->value.wval.w2)
#define SST_BEGG(p) ((p)->value.ilval.beg)
#define SST_ENDG(p) ((p)->value.ilval.end)
#define SST_COUNTG(p) ((p)->value.ilval.count)
#define SST_CLBEGG(p) ((p)->value.clval.beg)
#define SST_CLENDG(p) ((p)->value.clval.end)
#define SST_DBEGG(p) ((p)->value.dtval.beg)
#define SST_DENDG(p) ((p)->value.dtval.end)
#define SST_VLBEGG(p) ((p)->value.vlval.beg)
#define SST_VLENDG(p) ((p)->value.vlval.end)
#define SST_RNG1G(p) ((p)->value.wval.w1)
#define SST_RNG2G(p) ((p)->value.wval.w2)
#define SST_E1G(p) ((p)->value.tlval.e1)
#define SST_E2G(p) ((p)->value.tlval.e2)
#define SST_E3G(p) ((p)->value.tlval.e3)
/* for parsing acc routine */
#define SST_ROUTG(p) ((p)->value.wval.w3)
#define SST_DEVTYPEG(p) ((p)->value.wval.w2)
#define SST_DEVICEG(p) ((p)->value.wval.w2)

#define SST_OPTYPEP(p, v) ((p)->value.wval.w1 = (v))
#define SST_SYMP(p, v) ((p)->value.wval.w1 = (v))
#define SST_CVALP(p, v) ((p)->value.wval.w1 = (v))
#define SST_GDTYPEP(p, v) ((p)->value.wval.w2 = (v))
#define SST_DTYPEP(p, v) ((p)->value.wval.w2 = (v))
#define SST_GTYP(p, v) ((p)->value.wval.w3 = (v))
#define SST_LSYMP(p, v) ((p)->value.wval.w3 = (v))
#define SST_LENP(p, v) ((p)->value.wval.w3 = (v))
#define SST_ERRSYMP(p, v) ((p)->value.wval.w3 = (v))
#define SST_SHAPEP(p, v) ((p)->value.wval.w4 = (v))
#define SST_OPCP(p, v) ((p)->value.wval.w4 = (v))
#define SST_UNITP(p, v) ((p)->value.wval.w4 = (v))
#define SST_FIRSTP(p, v) ((p)->value.wval.w4 = (v))
#define SST_LASTP(p, v) ((p)->value.wval.w5 = (v))
#define SST_CVLENP(p, v) ((p)->value.wval.w5 = (v))
#define SST_CPLXPARTP(p, v) ((p)->value.wval.w5 = (v))
#define SST_ACLP(p, v) ((p)->value.cnval.acl = (v))
#define SST_SUBSCRIPTP(p, v) ((p)->value.eqvval.subscript = (v))
#define SST_SUBSTRINGP(p, v) ((p)->value.eqvval.substring = (v))
#define SST_OFFSETP(p, v) ((p)->value.eqvval.offset = (v))
#define SST_NMLBEGP(p, v) ((p)->value.wval.w1 = (v))
#define SST_NMLENDP(p, v) ((p)->value.wval.w2 = (v))
#define SST_BEGP(p, v) ((p)->value.ilval.beg = (v))
#define SST_ENDP(p, v) ((p)->value.ilval.end = (v))
#define SST_COUNTP(p, v) ((p)->value.ilval.count = (v))
#define SST_CLBEGP(p, v) ((p)->value.clval.beg = (v))
#define SST_CLENDP(p, v) ((p)->value.clval.end = (v))
#define SST_DBEGP(p, v) ((p)->value.dtval.beg = (v))
#define SST_DENDP(p, v) ((p)->value.dtval.end = (v))
#define SST_VLBEGP(p, v) ((p)->value.vlval.beg = (v))
#define SST_VLENDP(p, v) ((p)->value.vlval.end = (v))
#define SST_RNG1P(p, v) ((p)->value.wval.w1 = (v))
#define SST_RNG2P(p, v) ((p)->value.wval.w2 = (v))
#define SST_E1P(p, v) ((p)->value.tlval.e1 = (v))
#define SST_E2P(p, v) ((p)->value.tlval.e2 = (v))
#define SST_E3P(p, v) ((p)->value.tlval.e3 = (v))
/* for parsing acc routine */
#define SST_ROUTP(p, v) ((p)->value.wval.w3 = (v))
#define SST_DEVTYPEP(p, v) ((p)->value.wval.w2 = (v))
#define SST_DEVICEP(p, v) ((p)->value.wval.w2 = (v))

#define SST_ISNONDECC(p)    \
  (SST_IDG(p) == S_CONST && \
   (SST_DTYPEG(p) == DT_WORD || SST_DTYPEG(p) == DT_HOLL))

/* Functions that would be declared in semant.h but have SST in their
 * signatures are declared here instead.
 */

void semant1(int rednum, SST *top); /* semant.c */
void semant2(int rednum, SST *top); /* semant2.c */
void semant3(int rednum, SST *top); /* semant3.c */

void psemant1(int rednum, SST *top);  /* psemant.c */
void psemant2(int rednum, SST *top);  /* psemant2.c */
void psemant3(int rednum, SST *top);  /* psemant3.c */
void psemantio(int rednum, SST *top); /* psemantio.c */
void psemsmp(int rednum, SST *top);   /* psemsmp.c */
void semantio(int rednum, SST *top);  /* semantio.c */

/* semfunc.c */
int func_call2(SST *stktop, ITEM *list, int flag);
int func_call(SST *stktop, ITEM *list);
int ptrfunc_call(SST *stktop, ITEM *list);
void subr_call2(SST *stktop, ITEM *list, int flag);
void subr_call(SST *stktop, ITEM *list);
void ptrsubr_call(SST *stktop, ITEM *list);
void cuda_call(SST *stktop, ITEM *list, ITEM *chevlist);
int ref_intrin(SST *stktop, ITEM *list);
int ref_pd(SST *stktop, ITEM *list);

/* semfunc2.c */
int define_stfunc(int sptr, ITEM *argl, SST *estk);
int ref_stfunc(SST *stktop, ITEM *args);
int mkarg(SST *stkptr, int *dtype);
int chkarg(SST *stkptr, int *dtype);
int tempify(SST *stkptr);

/* semutil.c */
void constant_lvalue(SST *);
INT chkcon(SST *, int, LOGICAL);
ISZ_T chkcon_to_isz(SST *, LOGICAL);
INT chktyp(SST *, int, LOGICAL);
INT chk_scalartyp(SST *, int, LOGICAL);
INT chk_scalar_inttyp(SST *, int, const char *);
INT chk_arr_extent(SST *, const char *);
INT chksubscr(SST *, int);
int casttyp(SST *, int);
void cngtyp(SST *, DTYPE);
void cngshape(SST *, SST *);
LOGICAL chkshape(SST *, SST *, LOGICAL);
int chklog(SST *);
void mkident(SST *);
int mkexpr(SST *);
int mkexpr1(SST *);
int mkexpr2(SST *);
void mklogint4(SST *);
int mklvalue(SST *, int);
int mkvarref(SST *, ITEM *);
LOGICAL is_sst_const(SST *);
INT get_sst_cval(SST *);
LOGICAL is_varref(SST *);
int chksubstr(SST *, ITEM *);
void ch_substring(SST *, SST *, SST *);
int fix_term(SST *, int);
int assign(SST *, SST *);
int assign_pointer(SST *, SST *);
void chkopnds(SST *, SST *, SST *);
void unop(SST *, SST *, SST *);
void binop(SST *, SST *, SST *, SST *);
const char *prtsst(SST *);
int mklabelvar(SST *);

/* semutil2.c */
void construct_acl_for_sst(SST *, DTYPE);
void dinit_struct_param(SPTR, ACL *, DTYPE);
VAR *dinit_varref(SST *);
int sem_tempify(SST *);
int check_etmp(SST *);

/* semsmp.c */
void semsmp(int rednum, SST *top);
int mk_storage(int sptr, SST *stkp);
extern LOGICAL validate_omp_atomic(SST*, SST*);
extern int do_openmp_atomics(SST*, SST*);

/* semgnr.c */
int generic_tbp_call(int gnr, SST *stktop, ITEM *list, ITEM *chevlist);
void generic_call(int gnr, SST *stktop, ITEM *list, ITEM *chevlist);
int generic_tbp_func(int gnr, SST *stktop, ITEM *list);
int generic_func(int gnr, SST *stktop, ITEM *list);
int defined_operator(int opr, SST *stktop, SST *lop, SST *rop);
LOGICAL is_intrinsic_opr(int val, SST *stktop, SST *lop, SST *rop,
                         int tkn_alias);
int resolve_defined_io(int read_or_write, SST *stktop, ITEM *list);
