/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief SCFTN routine to process data initialization statements; called by
 * semant.
 */

#include "dinit.h"
#include "dinitutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "ilm.h"
#include "ilmtp.h"
#include "machardf.h"
#include "semutil0.h"
#include "symfun.h"

/** \brief Effective address of a reference being initialized */
typedef struct {
  SPTR sptr; /**< the containing object being initialized */
  SPTR mem;  /**< the variable or member being initialized; if not a member,
              * same as sptr.
              */
  ISZ_T offset;
} EFFADR;

typedef struct {
  int sptr;
  ISZ_T currval;
  ISZ_T upbd;
  ISZ_T step;
} DOSTACK;

static EFFADR *mkeffadr(int);
static ISZ_T eval(int);
static char *acl_idname(int);
static void dinit_subs(CONST *, SPTR, ISZ_T, SPTR);
static void dinit_val(SPTR sptr, DTYPE dtypev, INT val);
static ISZ_T get_ival(DTYPE, INT);
static INT _fdiv(INT dividend, INT divisor);
static void _ddiv(INT *dividend, INT *divisor, INT *quotient);
static CONST *eval_init_expr_item(CONST *cur_e);
static CONST *eval_array_constructor(CONST *e);
static CONST *eval_init_expr(CONST *e);
static CONST *eval_do(CONST *ido);
static CONST *clone_init_const(CONST *original, int temp);
static CONST *clone_init_const_list(CONST *original, int temp);
static void add_to_list(CONST *val, CONST **root, CONST **tail);
static void save_init(CONST *ict, SPTR sptr);
static void df_dinit(VAR *, CONST *);
static CONST *dinit_varref(VAR *ivl, SPTR member, CONST *ict, DTYPE dtype,
                           int *struct_bytes_initd, ISZ_T *repeat,
                           ISZ_T base_off);

static CONST **init_const; /* list of pointers to saved COSNT lists */
static int cur_init = 0;
int init_list_count = 0; /* size of init_const */
static CONST const_err;

#define CONST_ERR(dt) (const_err.dtype = dt, clone_init_const(&const_err, true))

static int substr_len; /* length of char substring being init'd */

#define MAXDIMS 7
#define MAXDEPTH 8
static DOSTACK dostack[MAXDEPTH];
static DOSTACK *top;
static DOSTACK *bottom;
static FILE *df; /* defer dinit until semfin */

/* Define repeat value when use of REPEAT dinit records becomes worthwhile */

#define THRESHOLD 6
#define MAX_EXP_QVALUE 4931 /* max value exponent */
#define MAX_EXP_OF_QMANTISSA 33 /* 2^112 : 5.19e+33 */
#define REAL_16 16
#define REAL_0 0
#define RADIX2 2 /* Binary */
#define NOT_GET_VAL 0 /* 0 is default, This should be ruled out */
#define NO_REAL -5 /* if the processor supports no real type with radix RADIX  */

/*****************************************************************/

/*
 * Instead of creating dinit records during the processing of data
 * initializations, we need to save information so the records are written
 * at the end of semantic analysis (during semfin).  This is necessary for
 * at least a couple of reasons: 1). a record dcl with inits in its
 * STRUCTURE
 * could occur before resolution of its storage class (problematic is
 * SC_CMBLK)  2). with VMS ftn, an array may be initialized (not by implied
 * DO) before resolution of its stype (i.e., its DIMENSION).
 *
 * The information we need to save is the pointers to the var list and
 * constant tree and the ilms.  This also implies that the getitem areas
 * (4, 5) need to stay around until semfin.
 */
void
dinit(VAR *ivl, CONST *ict)
{
  int nw;
  char *ptr;
  ILM_T *p;

  if (df == NULL) {
    if ((df = tmpfile()) == NULL)
      errfatal(F_0005_Unable_to_open_temporary_file);
  }
  ptr = (char *)ivl;
  nw = fwrite(&ptr, sizeof(ivl), 1, df);
  if (nw != 1)
    error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0,
          "(data init file)", CNULL);
  ptr = (char *)ict;
  nw = fwrite(&ptr, sizeof(ict), 1, df);
  if (nw != 1)
    error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0,
          "(data init file)", CNULL);
  p = ilmb.ilm_base;
  *p++ = IM_BOS;
  *p++ = gbl.lineno;
  *p++ = gbl.findex;
  *p = ilmb.ilmavl;
  nw = fwrite((char *)ilmb.ilm_base, sizeof(ILM_T), ilmb.ilmavl, df);
  if (nw != ilmb.ilmavl)
    error(F_0010_File_write_error_occurred_OP1, ERR_Fatal, 0,
          "(data init file)", CNULL);
#if DEBUG
  if (DBGBIT(6, 16)) {
    fprintf(gbl.dbgfil, "---- deferred dinit write: ivl %p, ict %p\n",
            (void *)ivl, (void *)ict);
    dumpilms();
  }
#endif

}

/*****************************************************************/

/**
   \brief a symbol is being initialized
   update certain attributes of the symbol including its dinit flag
 */
static void
sym_is_dinitd(SPTR sptr)
{
  DINITP(sptr, 1);
  if (SCG(sptr) == SC_CMBLK)
    /*  set DINIT flag for common block:  */
    DINITP(MIDNUMG(sptr), 1);

  /* For identifiers the DATA statement ensures that the identifier
   * is a variable and not an intrinsic.  For arrays, either
   * compute the element offset or if a whole array reference
   * compute the number of elements to initialize.
   */
  if (STYPEG(sptr) == ST_IDENT || STYPEG(sptr) == ST_UNKNOWN)
    STYPEP(sptr, ST_VAR);

}

void
do_dinit(void)
{
  /*
   * read in the information a "record" (2 pointers and ilms) at a time
   * saved by dinit(), and write dinit records for each record.
   */
  VAR *ivl;
  CONST *ict;
  char *ptr;
  int nw;
  int nilms;

  if (df == NULL)
    return;
  nw = fseek(df, 0L, 0);
#if DEBUG
  assert(nw == 0, "do_dinit:bad rewind", nw, ERR_Fatal);
#endif

  /* allocate the list of pointers to save initializer constant lists */
  init_const = (CONST **)getitem(4, init_list_count * sizeof(CONST *));
  BZERO(init_const, sizeof(CONST *), init_list_count);

  while (true) {
    nw = fread(&ptr, sizeof(ivl), 1, df);
    if (nw == 0)
      break;
#if DEBUG
    assert(nw == 1, "do_dinit: ict error", nw, ERR_Fatal);
#endif
    ivl = (VAR *)ptr;
    nw = fread(&ptr, sizeof(ict), 1, df);
#if DEBUG
    assert(nw == 1, "do_dinit: ivl error", nw, ERR_Fatal);
#endif
    ict = (CONST *)ptr;
    nw = fread((char *)ilmb.ilm_base, sizeof(ILM_T), BOS_SIZE, df);
#if DEBUG
    assert(nw == BOS_SIZE, "do_dinit: BOS error", nw, ERR_Fatal);
#endif
    /*
     * determine the number of words remaining in the ILM block
     */
    nilms = *(ilmb.ilm_base + 3);
    nw = nilms - BOS_SIZE;

    /* read in the remaining part of the ILM block  */

    nilms = fread((char *)(ilmb.ilm_base + BOS_SIZE), sizeof(ILM_T), nw, df);
#if DEBUG
    assert(nilms == nw, "do_dinit: BLOCK error", nilms, ERR_Severe);
#endif
    gbl.lineno = ilmb.ilm_base[1];
    gbl.findex = ilmb.ilm_base[2];
    ilmb.ilmavl = ilmb.ilm_base[3];
#if DEBUG
    if (DBGBIT(6, 32)) {
      fprintf(gbl.dbgfil, "---- deferred dinit read: ivl %p, ict %p\n",
              (void *)ivl, (void *)ict);
    }
#endif
    if (ict && ict->no_dinitp &&
        (SCG(ict->sptr) == SC_LOCAL || SCG(ict->sptr) == SC_PRIVATE))
      continue;
    df_dinit(ivl, ict);
  }

  fclose(df);
  df = NULL;
  freearea(5);

}

/**
 * \brief Find the sptr for the implied do index variable
 * The ilm in this context represents the ilms generated to load the index
 * variable and perhaps "type" convert (if it's integer*2, etc.).
 */
static int
chk_doindex(int ilmptr)
{
  int sptr;
again:
  switch (ILMA(ilmptr)) {
  case IM_I8TOI:
  case IM_STOI:
  case IM_SCTOI:
    ilmptr = ILMA(ilmptr + 1);
    goto again;
  case IM_KLD:
  case IM_ILD:
  case IM_SILD:
  case IM_CHLD:
    /* find BASE of load, and then sptr of BASE */
    sptr = ILMA(ILMA(ilmptr + 1) + 1);
    return sptr;
  }
  /* could use a better error message - illegal implied do index variable */
  errsev(S_0106_DO_index_variable_must_be_a_scalar_variable);
  sem.dinit_error = true;
  return 1;
}

/** \brief Initialize a data object
 *
 * \param ivl   pointer to initializer variable list
 * \param ict   pointer to initializer constant tree
 * \param dtype data type of structure type, if a struct init
 */
static void
dinit_data(VAR *ivl, CONST *ict, DTYPE dtype, ISZ_T base_off)
{
  SPTR member = SPTR_NULL;
  int struct_bytes_initd; /* use to determine fields in typedefs need
                           * to be padded */
  ILM_T *p;
  ISZ_T repeat = 0;

  if (ivl == NULL && dtype) {
    member = DTyAlgTyMember(DDTG(dtype));
    if (POINTERG(member)) {
      /* get to <ptr>$p */
      member = SYMLKG(member);
    }
    struct_bytes_initd = 0;
  }

  do {
    if (member) {
      if (POINTERG(member)) {
        /* get to <ptr>$p */
        member = SYMLKG(member);
      }
      if (is_empty_typedef(DTYPEG(member))) {
        member = SYMLKG(member);
        if (member == NOSYM)
          member = SPTR_NULL;
      }
    }
    if ((ivl && ivl->id == Varref) || member) {
      if (member && (CLASSG(member) && VTABLEG(member) &&
                     (TBPLNKG(member) || FINALG(member)))) {
        member = SYMLKG(member);
        if (member == NOSYM)
          member = SPTR_NULL;
        continue;
      } else
        ict = dinit_varref(ivl, member, ict, dtype, &struct_bytes_initd,
                           &repeat, base_off);
    } else if (ivl && ivl->id == Dostart) {
      if (top == &dostack[MAXDEPTH]) {
        /*  nesting maximum exceeded.  */
        errsev(S_0034_Syntax_error_at_or_near_OP1);
        return;
      }
      top->sptr = chk_doindex(ivl->u.dostart.indvar);
      if (top->sptr == 1)
        return;
      top->currval = eval(ivl->u.dostart.lowbd);
      top->upbd = eval(ivl->u.dostart.upbd);
      top->step = eval(ivl->u.dostart.step);

      if ((top->step > 0 && top->currval > top->upbd) ||
          (top->step <= 0 && top->currval < top->upbd)) {
        VAR *wivl;
        for (wivl = ivl; wivl->id != Doend && wivl->u.doend.dostart != ivl;
             wivl = wivl->next)
          ;

        ivl = wivl;
      } else {
        ++top;
      }
    } else if (ivl) {
      assert(ivl->id == Doend, "dinit:badid", 0, ERR_Severe);

      --top;
      top->currval += top->step;
      if ((top->step > 0 && top->currval <= top->upbd) ||
          (top->step <= 0 && top->currval >= top->upbd)) {
        /*  go back to start of this do loop  */
        ++top;
        ivl = ivl->u.doend.dostart;
      }
    }
    if (sem.dinit_error)
      goto error_exit;
    if (ivl)
      ivl = ivl->next;
    if (member) {
      struct_bytes_initd += size_of(DTYPEG(member));
      member = SYMLKG(member);
      if (POINTERG(member)) {
        /* get to <ptr>$p */
        member = SYMLKG(member);
      }
      if (member == NOSYM)
        member = SPTR_NULL;
    }
  } while (ivl || member);

/* Too many initializer is allowed.
if (ict)   errsev(67);
 */
 error_exit:;
#if DEBUG
  if (ivl && DBGBIT(6, 2) && ilmb.ilmavl != BOS_SIZE) {
    /* dump ilms afterwards because dmpilms overwrites opcodes */
    *(p = ilmb.ilm_base) = IM_BOS;
    *++p = gbl.lineno;
    *++p = gbl.findex;
    *++p = ilmb.ilmavl;
    dmpilms();
  }
#endif
}

/**
 * \param ivl pointer to initializer variable list
 * \param ict pointer to initializer constant tree
 */
static void
df_dinit(VAR *ivl, CONST *ict)
{
  CONST *new_ict;
#if DEBUG
  if (DBGBIT(6, 3)) {
    fprintf(gbl.dbgfil, "\nDINIT CALLED ----------------\n");
    if (DBGBIT(6, 2)) {
      if (ivl) {
        fprintf(gbl.dbgfil, "  Dinit Variable List:\n");
        dmp_ivl(ivl, gbl.dbgfil);
      }
      if (ict) {
        fprintf(gbl.dbgfil, "  Dinit Constant List:\n");
        dmp_ict(ict, gbl.dbgfil);
      }
    }
  }
#endif

  substr_len = 0;

  new_ict = eval_init_expr(ict);
#if DEBUG
  if (DBGBIT(6, 2)) {
    if (new_ict) {
      fprintf(gbl.dbgfil, "  Dinit new_Constant List:\n");
      dmp_ict(new_ict, gbl.dbgfil);
    }
  }
  if (DBGBIT(6, 1))
    fprintf(gbl.dbgfil, "\n  DINIT Records:\n");
#endif
  if (ivl) {
    sym_is_dinitd((SPTR)new_ict->sptr);
    bottom = top = &dostack[0];
    dinit_data(ivl, new_ict, DT_NONE, 0); /* Process DATA statements */
  } else {
    sym_is_dinitd((SPTR)ict->sptr);
    dinit_subs(new_ict, ict->sptr, 0, SPTR_NULL); /* Process type dcl inits and */
  }                                       /* init'ed structures */

#if DEBUG
  if (DBGBIT(6, 3))
    fprintf(gbl.dbgfil, "\nDINIT RETURNING ----------------\n\n");
#endif
}

/**
   \brief Return \c true if the constant of the given dtype represents zero
 */
static bool
is_zero(DTYPE dtype, INT conval)
{
  switch (DTY(dtype)) {
  case TY_INT8:
  case TY_LOG8:
    if (CONVAL2G(conval) == 0 && (!XBIT(124, 0x400) || CONVAL1G(conval) == 0))
      return true;
    break;
  case TY_INT:
  case TY_LOG:
  case TY_SINT:
  case TY_SLOG:
  case TY_BINT:
  case TY_BLOG:
  case TY_FLOAT:
    if (conval == 0)
      return true;
    break;
  case TY_DBLE:
    if (conval == stb.dbl0)
      return true;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    if (conval == stb.quad0)
      return true;
    break;
#endif
  case TY_CMPLX:
    if (CONVAL1G(conval) == 0 && CONVAL2G(conval) == 0)
      return true;
    break;
  case TY_DCMPLX:
    if (CONVAL1G(conval) == stb.dbl0 && CONVAL2G(conval) == stb.dbl0)
      return true;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    if (CONVAL1G(conval) == stb.quad0 && CONVAL2G(conval) == stb.quad0)
      return true;
    break;
#endif
  default:
    break;
  }
  return false;
}

static CONST *
dinit_varref(VAR *ivl, SPTR member, CONST *ict, DTYPE dtype,
             int *struct_bytes_initd, ISZ_T *repeat, ISZ_T base_off)
{
  SPTR sptr;      /* containing object being initialized */
  SPTR init_sym;  /* member or variable being initialized */
  ISZ_T offset, elsize, num_elem, i;
  bool new_block; /* flag to put out DINIT record */
  EFFADR *effadr; /* Effective address of array ref */
  bool zero;      /* is this put DINIT_ZEROES? */
  CONST *saved_ict;
  bool put_value = true;
  int ilmptr;

  if (ivl && ivl->u.varref.id == S_IDENT) {
    /* We are dealing with a scalar or whole array init */
    ilmptr = ivl->u.varref.ptr;
    /*
     * DINITPOINTER23995 - when POINTER dinits are passed thru, the reference
     * ILM  will be a IM_PLD -- its operand is an IM_BASE.
     */
    if (ILMA(ilmptr) == IM_PLD)
      ilmptr = ILMA(ilmptr+1);
    assert(ILMA(ilmptr) == IM_BASE, "dinit_data not IM_BASE", ilmptr, ERR_Severe);
    init_sym = sptr = (SPTR)ILMA(ilmptr + 1);
    if (!dinit_ok(sptr))
      goto error_exit;
    num_elem = 1;
    offset = 0;
    if (!POINTERG(sptr) && DTY(DTYPEG(sptr)) == TY_ARRAY) {
      /* A whole array so determine number of elements to init */
      if (extent_of(DTYPEG(sptr)))
        num_elem = ad_val_of(AD_NUMELM(AD_PTR(sptr)));
      else
        num_elem = 0;
      if (num_elem == 0)
        elsize = size_of(DTYPEG(sptr));
      else
        elsize = size_of(DTYPEG(sptr)) / num_elem;
    }
  } else if (member) {
    init_sym = sptr = member;
    num_elem = 1;
    offset = ADDRESSG(sptr) + base_off;
    elsize = size_of(DTYPEG(sptr));
    if (!POINTERG(sptr) && DTY(DTYPEG(sptr)) == TY_ARRAY) {
      /* A whole array so determine number of elements to init */
      if (extent_of(DTYPEG(sptr)))
        num_elem = ad_val_of(AD_NUMELM(AD_PTR(sptr)));
      else
        num_elem = 0;
      if (num_elem == 0)
        elsize = size_of(DTYPEG(sptr));
      else
        elsize = size_of(DTYPEG(sptr)) / num_elem;
    }
  } else {
    /* We are dealing with an array element, array slice,
     * character substr_len, or derived type member init.
     */
    /* First dereference the ilm ptr to a symbol pointer */
    effadr = mkeffadr(ivl->u.varref.ptr);
    if (sem.dinit_error)
      goto error_exit;
    if (ivl->u.varref.shape != 0)
      uf("array section");
    sptr = effadr->sptr;
    num_elem = 1;
    offset = effadr->offset;
    elsize = 1; /* doesn't matter since num_elem is 1 */
    init_sym = effadr->mem;
    if (sptr != init_sym && DTY(DTYPEG(init_sym)) == TY_ARRAY &&
        ILMA(ivl->u.varref.ptr) != IM_ELEMENT) {
      /* A whole array so determine number of elements to init */
      num_elem = ad_val_of(AD_NUMELM(AD_PTR(init_sym)));
      if (num_elem == 0)
        elsize = size_of(DTYPEG(sptr));
      else
        elsize = size_of(DTYPEG(init_sym)) / num_elem;
    }
  }

  /*  now process enough dinit constant list items to
      take care of the current varref item:  */
  new_block = true;
  saved_ict = ict;

/* if this symbol is defined in an outer scope or
 *    the symbol is a member of a common block
      not defined in this procedure (i.e., DINITG not set)
 *  then plug the symbol table with the initializer list but
 *   don't write the values to the dinit file becasue it has already been done
 */
  if (UPLEVELG(sptr) || (SCG(sptr) == SC_CMBLK && !DINITG(sptr))) {
    put_value = false;
  }

  if (ict && *repeat == 0) {
    *repeat = ict->repeatc;
  }
  do {
    if (no_data_components(DDTG(DTYPEG(sptr))) &&
        !is_zero_size_typedef(DDTG(DTYPEG(sptr)))) {
      break;
    }
    if (ict == NULL) {
      errsev(S_0066_Too_few_data_constants_in_initialization_statement);
      goto error_exit;
    }

    if (ict->id == AC_ACONST) {
      *repeat = 0;
      (void)dinit_varref(ivl, member, ict->subc, dtype, struct_bytes_initd,
                         repeat, base_off);
      /* Make sure the recursive processing ends, as the nesting has been
       * collapsed/flattened in eval_array_constructor */
      if (ict->subc->subc != nullptr && ict->u1.ido.index_var != 0) {
        interr("nested implied do loop should have been collapsed/flattened", 0, ERR_Warning);
      }
      *repeat = num_elem;
      i = num_elem;
    } else {
      if (ivl && (DTY(DDTG(ivl->u.varref.dtype)) == TY_STRUCT)) {
        if (put_value) {
          if (base_off == 0) {
            dinit_put(DINIT_LOC, (ISZ_T)sptr);
          }
          if (DTY(DTYPEG(sptr)) == TY_ARRAY && offset) {
            dinit_put(DINIT_OFFSET, offset);
            dinit_data(NULL, ict->subc, ict->dtype, 0);
          } else if (is_zero_size_typedef(DDTG(ivl->u.varref.dtype))) {
            dinit_put(DINIT_OFFSET, offset);
          } else {
            dinit_data(NULL, ict->subc, ict->dtype, offset);
          }
        }
        i = 1;
        new_block = true;
      } else if (member && DTY(ict->dtype) == TY_STRUCT) {
        if (put_value) {
          dinit_data(NULL, ict->subc, ict->dtype, offset);
        }
        i = 1;
        new_block = true;
      } else {
        /* if there is a repeat count in the data item list,
         * only use as many as in this array */
        i = (num_elem < *repeat) ? num_elem : *repeat;
        if (i < THRESHOLD)
          i = 1;
        if (ivl == NULL && member)
          i = 1;
        zero = false;
        if (put_value) {
          if (new_block || i != 1) {
            if (!member)
              dinit_put(DINIT_LOC, (ISZ_T)sptr);
            if (offset)
              dinit_put(DINIT_OFFSET, offset);
            if (i != 1) {
              if (i > 1 && is_zero(ict->dtype, ict->u1.conval)) {
                dinit_put(DINIT_ZEROES, i * elsize);
                zero = true;
              } else {
                dinit_put(DINIT_REPEAT, (ISZ_T)i);
              }
              new_block = true;
            } else {
              new_block = false;
            }
          }
          if (!zero) {
            if (DTY(ict->dtype) == TY_STRUCT) {
              dinit_data(NULL, ict->subc, ict->dtype, base_off);
            } else {
              dinit_val(init_sym, ict->dtype, ict->u1.conval);
            }
          }
        }
      }
    }
    offset += i * elsize;
    num_elem -= i;
    *repeat -= i;
    if (*repeat == 0) {
      ict = ict->next;
      *repeat = ict ? ict->repeatc : 0;
    }
  } while (num_elem > 0);
  if (put_value) {
    sym_is_dinitd(sptr);
  }

  if ((!member && PARAMG(sptr)) || (CCSYMG(sptr) && DINITG(sptr))) {
    /* this variable may be used in other initializations,
     * save its initializer list
     */
    save_init(clone_init_const_list(saved_ict, false), sptr);
  }

  return ict;

error_exit:
  sem.dinit_error++;
  return NULL;
}

/**
   \brief FIXME
   \param ict      pointer to initializer constant tree
   \param base     sym pointer to base address
   \param boffset  current offset from base
   \param mbr_sptr sptr of member if processing typedef
 */
static void
dinit_subs(CONST *ict, SPTR base, ISZ_T boffset, SPTR mbr_sptr)
{
  ISZ_T loffset = 0; /*offset from begin of current structure */
  ISZ_T roffset = 0; /* offset from begin of member (based on repeat count) */
  ISZ_T toffset = 0; /* temp offset of for roffset, set it back to previous
                        roffset after dinit_subs call */
  SPTR sptr;         /* symbol ptr to identifier to get initialized */
  SPTR sub_sptr;     /* sym ptr to nested type/struct fields */
  ISZ_T i;
  DTYPE dtype;       /* data type of member being initialized */
  ISZ_T elsize = 0;  /* size of basic or array element in bytes */
  ISZ_T num_elem;    /* if handling an array, number of array elements else 1 */
  bool new_block;    /* flag indicating need for DINIT_LOC record.  Always
                      * needed after a DINIT_REPEAT block */

  /*
   * We come into this routine to follow the ict links for a substructure.
   * 'boffset' comes in as the offset from the beginning of the parent
   * structure for the structure we are going to traverse.
   *
   * We have two other offsets while traversing this structure.  'loffset'
   * is the local offset from the beginning of this structure.  'roffset'
   * is the offset based on repeat counts.
   */
  new_block = true;
  while (ict) {
    if (ict->subc) {
      /* Follow substructure down before continuing at this level */
      roffset = 0;
      loffset = 0;
      num_elem = 1;
      if (ict->id == AC_SCONST) {
        if (ict->sptr) {
          sub_sptr = DTyAlgTyMember(DDTG(DTYPEG(ict->sptr)));
          if (mbr_sptr) {
            loffset = ADDRESSG(ict->sptr);
          }
        } else if (mbr_sptr) {
          dtype = DDTG(DTYPEG(mbr_sptr));
          sub_sptr = (DTY(dtype) == TY_STRUCT)
            ? DTyAlgTyMember(DDTG(DTYPEG(mbr_sptr))) : mbr_sptr;
          loffset = ADDRESSG(mbr_sptr);
          if (DTY(DTYPEG(mbr_sptr)) == TY_ARRAY) {
            num_elem = ad_val_of(AD_NUMELM(AD_DPTR(DTYPEG(mbr_sptr))));
          }
        } else {
          interr("dinit_subs: malformed derived type init,"
                 " unable to determine member for", base, ERR_Severe);
          return;
        }
      } else if (ict->id == AC_ACONST) {
        if (ict->sptr) {
          sub_sptr = ict->sptr;
        } else if (mbr_sptr) {
          sub_sptr = mbr_sptr;
        } else {
          interr("dinit_subs: malformed  array init,"
                 " unable to determine member for",
                 base, ERR_Severe);
          return;
        }
      } else {
        sub_sptr = SPTR_NULL;
      }

      /* per flyspray 15963, the roffset must be set back to its value
       * before a call to dinit_subs in for loop.
       */
      toffset = roffset;
      for (i = ict->repeatc; i != 0; i--) {
        dinit_subs(ict->subc, base, boffset + loffset + roffset, sub_sptr);
        roffset += DTyAlgTySize(ict->dtype);
      }
      roffset = toffset;
      num_elem -= ict->repeatc;
      ict = ict->next;
      new_block = true;
    } else {
      /* Handle basic type declaration init statement */
      /* If new member or member has a repeat start a new block */
      if (ict->sptr) {
        /* A new member to initialize */
        sptr = ict->sptr;
        roffset = 0;
        loffset = ADDRESSG(sptr);
        dtype = (DTYPEG(sptr));
        elsize = size_of(dtype);
        if (DTY(dtype) == TY_ARRAY)
          elsize /= ad_val_of(AD_NUMELM(AD_PTR(sptr)));
        new_block = true;
      } else {
        if (ict->repeatc > 1) {
          new_block = true;
        }
        if (mbr_sptr) {
          sptr = mbr_sptr;
          dtype = (DTYPEG(sptr));
          loffset = ADDRESSG(mbr_sptr);
          roffset = 0;
          elsize = size_of(dtype);
          if (DTY(dtype) == TY_ARRAY)
            elsize /= ad_val_of(AD_NUMELM(AD_PTR(sptr)));
        }
      }
      if (new_block) {
        dinit_put(DINIT_LOC, (ISZ_T)base);
        dinit_put(DINIT_OFFSET, boffset + loffset + roffset);
        new_block = false;
      }
      if (ict->repeatc > 1) {
        new_block = true;
        dinit_put(DINIT_REPEAT, (ISZ_T)ict->repeatc);
        num_elem = 1;
      } else {
        num_elem =
            (DTY(dtype) == TY_ARRAY) ? ad_val_of(AD_NUMELM(AD_DPTR(dtype))) : 1;
      }
      roffset += elsize * ict->repeatc;

      do {
        dinit_val(sptr, ict->dtype, ict->u1.conval);
        ict = ict->next;
      } while (--num_elem > 0);
    }
    if (ict && mbr_sptr) {
      if (ict->sptr) {
        mbr_sptr = ict->sptr;
      } else if (num_elem <= 0) {
        mbr_sptr = SYMLKG(mbr_sptr);
      }
      if (mbr_sptr == NOSYM) {
        mbr_sptr = SPTR_NULL;
      } else {
        new_block = true;
      }
    }
  } /* End of while */
}

/*****************************************************************/
/* dinit_val - make sure constant value is correct data type to initialize
 * symbol (sptr) to.  Then call dinit_put to generate dinit record.
 */
static void
dinit_val(SPTR sptr, DTYPE dtypev, INT val)
{
  DTYPE dtype;
  char buf[2];

  dtype = (dtypev == DINIT_PROC) ? DINIT_PROC : DDTG(DTYPEG(sptr));
  if (no_data_components(dtype)) {
    return;
  }

  if (substr_len) {
/*
 * since substr_len is non-zero, it was specified in a substring
 * operation; dtype is modified to reflect this length instead
 * of the symbol's declared length.
 */
    TY_KIND dty = DTY(dtype);
    assert(dty == TY_CHAR || dty == TY_NCHAR, "dinit_val:nonchar sym", sptr,
           ERR_Severe);
    dtype = get_type(2, dty, substr_len);
    substr_len = 0;
  }

  if (DTYG(dtypev) == TY_HOLL) {
    /* convert hollerith character string to one of proper length */
    val = cngcon(val, DTYPEG(val), dtype);
  } else if ((DTYG(dtypev) == TY_CHAR || DTYG(dtypev) == TY_NCHAR ||
             DTYG(dtypev) != DTY(dtype)) &&
             !(POINTERG(sptr) && val == 0 && dtypev == DT_INT)) {
    /*  check for special case of initing character*1 to  numeric. */
    if (DTY(dtype) == TY_CHAR && DTyCharLength(dtype) == 1) {
      if (DT_ISINT(dtypev) && !DT_ISLOG(dtypev)) {
        if (flg.standard)
          error(W_0172_F77_extension_numeric_initialization_of_CHARACTER_OP1, ERR_Warning, gbl.lineno, SYMNAME(sptr), CNULL);
        if (val < 0 || val > 255) {
          error(S_0068_Numeric_initializer_for_CHARACTER_OP1_out_of_range_0_through_255, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL);
          val = getstring(" ", 1);
        } else {
          buf[0] = (char)val;
          buf[1] = 0;
          val = getstring(buf, 1);
        }
        dtypev = DT_CHAR;
      }
    }
    /* Convert character string to one of proper length or,
     * convert constant to type of identifier
     */
    val = cngcon(val, dtypev, dtype);
  }
  dinit_put(dtype, val);

  if (flg.opt >= 2 && STYPEG(sptr) == ST_VAR && SCG(sptr) == SC_LOCAL) {
    NEED(aux.dvl_avl + 1, aux.dvl_base, DVL, aux.dvl_size, aux.dvl_size + 32);
    DVL_SPTR(aux.dvl_avl) = sptr;
    DVL_CONVAL(aux.dvl_avl) = val;
    REDUCP(sptr, 1); /* => in dvl table */
    aux.dvl_avl++;
  }

}

/*****************************************************************/

void
dmp_ivl(VAR *ivl, FILE *f)
{
  FILE *dfil;
  dfil = f ? f : stderr;
  while (ivl) {
    if (ivl->id == Dostart) {
      fprintf(dfil, "    Do begin marker  (0x%p):", (void *)ivl);
      fprintf(dfil, " indvar: %4d lowbd:%4d", ivl->u.dostart.indvar,
              ivl->u.dostart.lowbd);
      fprintf(dfil, " upbd:%4d  step:%4d\n", ivl->u.dostart.upbd,
              ivl->u.dostart.step);
    } else if (ivl->id == Varref) {
      fprintf(dfil, "    Variable reference (");
      if (ivl->u.varref.id == S_IDENT) {
        fprintf(dfil, " S_IDENT):");
        fprintf(dfil, " sptr: %d(%s)", ILMA(ivl->u.varref.ptr + 1),
                SYMNAME(ILMA(ivl->u.varref.ptr + 1)));
        fprintf(dfil, " dtype: %4d\n", DTYPEG(ILMA(ivl->u.varref.ptr + 1)));
      } else {
        fprintf(dfil, "S_LVALUE):");
        fprintf(dfil, "  ilm:%4d", ivl->u.varref.ptr);
        fprintf(dfil, " shape:%4d\n", ivl->u.varref.shape);
      }
    } else {
      assert(ivl->id == Doend, "dmp_ivl: badid", 0, ERR_Severe);
      fprintf(dfil, "    Do end marker:");
      fprintf(dfil, "   Pointer to Do Begin: %p\n",
              (void *)ivl->u.doend.dostart);
    }
    ivl = ivl->next;
  }
}

void
dmp_ict(CONST *ict, FILE *f)
{
  FILE *dfil;
  dfil = f ? f : stderr;
  while (ict) {
    fprintf(dfil, "%p(%s):", (void *)ict, acl_idname(ict->id));
    if (ict->subc) {
      fprintf(dfil, "  subc: for structure tag %s  ",
              SYMNAME(DTyAlgTyTag(ict->dtype)));
      fprintf(dfil, "  sptr: %d", ict->sptr);
      if (ict->sptr) {
        fprintf(dfil, "(%s)", SYMNAME(ict->sptr));
      }
      fprintf(dfil, "  mbr: %d", ict->mbr);
      if (ict->mbr) {
        fprintf(dfil, "(%s)", SYMNAME(ict->mbr));
      }
      fprintf(dfil, "  rc: %" ISZ_PF "d", ict->repeatc);
      /*fprintf(dfil, "  next:%p\n", (void *)(ict->next));*/
      fprintf(dfil, "\n");
      dmp_ict(ict->subc, f);
      fprintf(dfil, "    Back from most recent substructure %p\n", ict);
      ict = ict->next;
    } else {
      fprintf(dfil, "  val: %6d   dt: %4d   rc: %6" ISZ_PF "d", ict->u1.conval,
              ict->dtype, ict->repeatc);
      fprintf(dfil, "  sptr: %d", ict->sptr);
      if (ict->sptr) {
        fprintf(dfil, "(%s)", SYMNAME(ict->sptr));
      }
      fprintf(dfil, "  mbr: %d", ict->mbr);
      if (ict->mbr) {
        fprintf(dfil, "(%s)", SYMNAME(ict->mbr));
      }
      /*fprintf(dfil, "  next:%p\n", (void *)(ict->next));*/
      fprintf(dfil, "\n");
      ict = ict->next;
    }
  }
}

static char *
acl_idname(int id)
{
  static char bf[32];
  switch (id) {
  case AC_IDENT:
    strcpy(bf, "IDENT");
    break;
  case AC_CONST:
    strcpy(bf, "CONST");
    break;
  case AC_EXPR:
    strcpy(bf, "EXPR");
    break;
  case AC_IEXPR:
    strcpy(bf, "IEXPR");
    break;
  case AC_AST:
    strcpy(bf, "AST");
    break;
  case AC_IDO:
    strcpy(bf, "IDO");
    break;
  case AC_REPEAT:
    strcpy(bf, "REPEAT");
    break;
  case AC_ACONST:
    strcpy(bf, "ACONST");
    break;
  case AC_SCONST:
    strcpy(bf, "SCONST");
    break;
  case AC_LIST:
    strcpy(bf, "LIST");
    break;
  case AC_VMSSTRUCT:
    strcpy(bf, "VMSSTRUCT");
    break;
  case AC_VMSUNION:
    strcpy(bf, "VMSUNION");
    break;
  case AC_TYPEINIT:
    strcpy(bf, "TYPEINIT");
    break;
  case AC_ICONST:
    strcpy(bf, "ICONST");
    break;
  case AC_CONVAL:
    strcpy(bf, "CONVAL");
    break;
  case AC_TRIPLE:
    strcpy(bf, "TRIPLE");
    break;
  default:
    sprintf(bf, "UNK_%d", id);
    break;
  }
  return bf;
}

/*****************************************************************/

/** \brief derefence an ilm pointer to determine the effective address of a
 *  reference (i.e. base sptr + byte offset).
 */
static EFFADR *
mkeffadr(int ilmptr)
{
  EFFADR *effadr;
  ADSC *ad;          /* Ptr to array descriptor */
  static EFFADR buf; /* Area ultimately returned containing effective addr */
  int i;
  ISZ_T offset, totoffset;
  ISZ_T lwbd;

  int opr1 = ILMA(ilmptr + 1);
  int opr2 = ILMA(ilmptr + 2);

  switch (ILMA(ilmptr)) {
  case IM_SUBS:
  case IM_NSUBS:
    effadr = mkeffadr(opr1);
    if (sem.dinit_error)
      break;
    lwbd = eval(opr2);
    if (ILMA(ilmptr) == IM_NSUBS) /* NCHAR/kanji - 2 bytes per char */
      effadr->offset += 2 * (lwbd - 1);
    else
      effadr->offset += lwbd - 1;
    /*  for kanji, substr_len in units of chars, not bytes: */
    substr_len = eval((int)ILMA(ilmptr + 3)) - lwbd + 1;
    break;

  case IM_ELEMENT:
    effadr = mkeffadr(opr2);
    if (sem.dinit_error)
      break;
    ad = AD_PTR(effadr->mem);
    totoffset = 0;
    for (i = 0; i < opr1; i++) {
      lwbd = ad_val_of(AD_LWBD(ad, i));
      offset = eval(ILMA(ilmptr + 4 + i));
      if (offset < lwbd || offset > ad_val_of(AD_UPBD(ad, i))) {
        error(S_0080_Subscript_for_array_OP1_is_out_of_bounds, ERR_Severe, gbl.lineno, SYMNAME(effadr->sptr), CNULL);
        sem.dinit_error = true;
        break;
      }
      totoffset += (offset - lwbd) * ad_val_of(AD_MLPYR(ad, i));
    }
    /* Convert array element offset to a byte offset */
    totoffset *= size_of(DDTG(DTYPEG(effadr->mem)));
    effadr->offset += totoffset;
    break;

  case IM_BASE:
    effadr = &buf;
    if (!dinit_ok(opr1))
      break;
    effadr->sptr = effadr->mem = (SPTR)opr1;
    effadr->offset = 0;
    break;

  case IM_MEMBER:
    effadr = mkeffadr(opr1);
    if (sem.dinit_error)
      break;
    effadr->mem = (SPTR)opr2;
    effadr->offset += ADDRESSG(opr2);
    break;

  case IM_IFUNC:
  case IM_KFUNC:
  case IM_RFUNC:
  case IM_DFUNC:
  case IM_CFUNC:
  case IM_CDFUNC:
  case IM_CALL:
    effadr = &buf;
    effadr->sptr = effadr->mem = (SPTR)opr2;
    error(S_0076_Subscripts_specified_for_non_array_variable_OP1, ERR_Severe, gbl.lineno, SYMNAME(effadr->sptr), CNULL);
    sem.dinit_error = true;
    break;

  default:
    effadr = &buf;
    effadr->sptr = SPTR_NULL;
    effadr->mem = SPTR_NULL;
    effadr = &buf;
    sem.dinit_error = true;
    break;
  }
  return effadr;
}

/*****************************************************************/

static ISZ_T
eval(int ilmptr)
{
  int opr1 = ILMA(ilmptr + 1);
  DOSTACK *p;

  switch (ILMA(ilmptr) /* opc */) {
  case IM_KLD:
  case IM_ILD:
  case IM_SILD:
  case IM_CHLD:
    /*  see if this ident is an active do index variable: */
    opr1 = ILMA(opr1 + 1); /* get sptr from BASE ilm */
    for (p = bottom; p < top; p++)
      if (p->sptr == opr1)
        return p->currval;
    /*  else - illegal use of variable: */
    error(S_0064_Illegal_use_of_OP1_in_DATA_statement_implied_DO_loop, ERR_Severe, gbl.lineno, SYMNAME(opr1), CNULL);
    sem.dinit_error = true;
    return 1L;

  case IM_KCON:
    return get_isz_cval(opr1);

  case IM_ICON:
    return CONVAL2G(opr1);

  case IM_KNEG:
  case IM_INEG:
    return -eval(opr1);
  case IM_KADD:
  case IM_IADD:
    return eval(opr1) + eval(ILMA(ilmptr + 2));
  case IM_KSUB:
  case IM_ISUB:
    return eval(opr1) - eval(ILMA(ilmptr + 2));
  case IM_KMUL:
  case IM_IMUL:
    return eval(opr1) * eval(ILMA(ilmptr + 2));
  case IM_KDIV:
  case IM_IDIV:
    return eval(opr1) / eval(ILMA(ilmptr + 2));
  case IM_ITOI8:
  case IM_I8TOI:
  case IM_STOI:
  case IM_SCTOI:
    /* these should reference SILD/CHLD */
    return eval(opr1);

  default:
    errsev(S_0069_Illegal_implied_DO_expression);
    sem.dinit_error = true;
    return 1L;
  }
}

static ISZ_T
get_ival(DTYPE dtype, INT conval)
{
  switch (DTY(dtype)) {
  case TY_INT8:
  case TY_LOG8:
    return get_isz_cval(conval);
  default:
    break;
  }
  return conval;
}

/*****************************************************************/

/**
   \brief determine if the symbol can be legally data initialized
 */
bool
dinit_ok(int sptr)
{
  switch (SCG(sptr)) {
  case SC_DUMMY:
    error(W_0041_Illegal_use_of_dummy_argument_OP1, ERR_Severe, gbl.lineno, SYMNAME(sptr), CNULL);
    goto error_exit;
  case SC_BASED:
    error(S_0116_Illegal_use_of_pointer_based_variable_OP1_OP2, ERR_Severe, gbl.lineno, SYMNAME(sptr), "(data initialization)");
    goto error_exit;
  case SC_CMBLK:
    if (ALLOCG(MIDNUMG(sptr))) {
      error(S_0163_Cannot_data_initialize_member_OP1_of_ALLOCATABLE_COMMON_OP2, ERR_Severe, gbl.lineno, SYMNAME(sptr), SYMNAME(MIDNUMG(sptr)));
      goto error_exit;
    }
    break;
  default:
    break;
  }
  if (STYPEG(sptr) == ST_ARRAY) {
    if (ALLOCG(sptr)) {
      error(S_0084_Illegal_use_of_symbol_OP1_OP2, ERR_Severe, gbl.lineno, SYMNAME(sptr),
            "- initializing an allocatable array");
      goto error_exit;
    }
    if (ASUMSZG(sptr)) {
      error(S_0084_Illegal_use_of_symbol_OP1_OP2, ERR_Severe, gbl.lineno, SYMNAME(sptr),
            "- initializing an assumed size array");
      goto error_exit;
    }
    if (ADJARRG(sptr)) {
      error(S_0084_Illegal_use_of_symbol_OP1_OP2, ERR_Severe, gbl.lineno, SYMNAME(sptr),
            "- initializing an adjustable array");
      goto error_exit;
    }
  }

  return true;

error_exit:
  sem.dinit_error = true;
  return false;
}

static INT
_fdiv(INT dividend, INT divisor)
{
  INT quotient;

#ifdef TM_FRCP
  if (!flg.ieee) {
    INT temp;
    xfrcp(divisor, &temp);
    xfmul(dividend, temp, &quotient);
  } else {
    xfdiv(dividend, divisor, &quotient);
  }
#else
  xfdiv(dividend, divisor, &quotient);
#endif
  return quotient;
}

static void
_ddiv(INT *dividend, INT *divisor, INT *quotient)
{
#ifdef TM_DRCP
  INT temp[2];

  if (!flg.ieee) {
    xdrcp(divisor, temp);
    xdmul(dividend, temp, quotient);
  } else {
    xddiv(dividend, divisor, quotient);
  }
#else
  xddiv(dividend, divisor, quotient);
#endif
}

static int
get_ast_op(int op)
{
  int ast_op = -1;

  switch (op) {
  case AC_NEG:
    ast_op = OP_NEG;
    break;
  case AC_ADD:
    ast_op = OP_ADD;
    break;
  case AC_SUB:
    ast_op = OP_SUB;
    break;
  case AC_MUL:
    ast_op = OP_MUL;
    break;
  case AC_DIV:
    ast_op = OP_DIV;
    break;
  case AC_CAT:
    ast_op = OP_CAT;
    break;
  case AC_LEQV:
    ast_op = OP_LEQV;
    break;
  case AC_LNEQV:
    ast_op = OP_LNEQV;
    break;
  case AC_LOR:
    ast_op = OP_LOR;
    break;
  case AC_LAND:
    ast_op = OP_LAND;
    break;
  case AC_EQ:
    ast_op = OP_EQ;
    break;
  case AC_GE:
    ast_op = OP_GE;
    break;
  case AC_GT:
    ast_op = OP_GT;
    break;
  case AC_LE:
    ast_op = OP_LE;
    break;
  case AC_LT:
    ast_op = OP_LT;
    break;
  case AC_NE:
    ast_op = OP_NE;
    break;
  case AC_LNOT:
    ast_op = OP_LNOT;
    break;
  case AC_EXP:
    ast_op = OP_XTOI;
    break;
  case AC_EXPK:
    ast_op = OP_XTOK;
    break;
  case AC_EXPX:
    ast_op = OP_XTOX;
    break;
  default:
    interr("get_ast_op: unexpected operator in initialization expr", op, ERR_Severe);
  }
  return ast_op;
}

/* Routine init_fold_const stolen from file ast.c in Fortran frontend */
static INT
init_fold_const(int opr, INT conval1, INT conval2, DTYPE dtype)
{
#ifdef TARGET_SUPPORTS_QUADFP
  IEEE128 qtemp, qresult, qnum1, qnum2;
  IEEE128 qreal1, qreal2, qrealrs, qimag1, qimag2, qimagrs;
  IEEE128 qtemp1, qtemp2;
#endif
  DBLE dtemp, dresult, num1, num2;
  DBLE dreal1, dreal2, drealrs, dimag1, dimag2, dimagrs;
  DBLE dtemp1, dtemp2;
  SNGL temp, result;
  SNGL real1, real2, realrs, imag1, imag2, imagrs;
  SNGL temp1;
  DBLINT64 inum1, inum2, ires;
  INT val;
  int term, sign;
  int cvlen1, cvlen2;
  char *p, *q;

  if (opr == OP_XTOI) {
    term = 1;
    if (dtype != DT_INT)
      term = cngcon(term, DT_INT, dtype);
    val = term;
    if (conval2 >= 0)
      sign = 0;
    else {
      conval2 = -conval2;
      sign = 1;
    }
    while (conval2--)
      val = init_fold_const(OP_MUL, val, conval1, dtype);
    if (sign) {
      /* exponentiation to a negative power */
      val = init_fold_const(OP_DIV, term, val, dtype);
    }
    return val;
  }
  if (opr == OP_XTOK) {
    ISZ_T cnt;
    term = stb.k1;
    if (dtype != DT_INT8)
      term = cngcon(term, DT_INT8, dtype);
    val = term;
    cnt = get_isz_cval(conval2);
    if (cnt >= 0)
      sign = 0;
    else {
      cnt = -cnt;
      sign = 1;
    }
    while (cnt--)
      val = init_fold_const(OP_MUL, val, conval1, dtype);
    if (sign) {
      /* exponentiation to a negative power */
      val = init_fold_const(OP_DIV, term, val, dtype);
    }
    return val;
  }
  switch (DTY(dtype)) {
  default:
    break;
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    switch (opr) {
    case OP_ADD:
      return conval1 + conval2;
    case OP_CMP:
      if (conval1 < conval2)
        return (INT)-1;
      if (conval1 > conval2)
        return (INT)1;
      return (INT)0;
    case OP_SUB:
      return conval1 - conval2;
    case OP_MUL:
      return conval1 * conval2;
    case OP_DIV:
      if (conval2 == 0) {
        errsev(S_0098_Divide_by_zero);
        conval2 = 1;
      }
      return conval1 / conval2;
    }
    break;

  case TY_INT8:
    inum1[0] = CONVAL1G(conval1);
    inum1[1] = CONVAL2G(conval1);
    inum2[0] = CONVAL1G(conval2);
    inum2[1] = CONVAL2G(conval2);
    switch (opr) {
    case OP_ADD:
      add64(inum1, inum2, ires);
      break;
    case OP_CMP:
      return cmp64(inum1, inum2);
    case OP_SUB:
      sub64(inum1, inum2, ires);
      break;
    case OP_MUL:
      mul64(inum1, inum2, ires);
      break;
    case OP_DIV:
      if (inum2[0] == 0 && inum2[1] == 0) {
        errsev(S_0098_Divide_by_zero);
        inum2[1] = 1;
      }
      div64(inum1, inum2, ires);
      break;
    }
    return getcon(ires, DT_INT8);

  case TY_REAL:
    switch (opr) {
    case OP_ADD:
      xfadd(conval1, conval2, &result);
      return result;
    case OP_SUB:
      xfsub(conval1, conval2, &result);
      return result;
    case OP_MUL:
      xfmul(conval1, conval2, &result);
      return result;
    case OP_DIV:
      result = _fdiv(conval1, conval2);
      return result;
    case OP_CMP:
      return xfcmp(conval1, conval2);
    case OP_XTOX:
      xfpow(conval1, conval2, &result);
      return result;
    }
    break;

  case TY_DBLE:
    num1[0] = CONVAL1G(conval1);
    num1[1] = CONVAL2G(conval1);
    num2[0] = CONVAL1G(conval2);
    num2[1] = CONVAL2G(conval2);
    switch (opr) {
    case OP_ADD:
      xdadd(num1, num2, dresult);
      break;
    case OP_SUB:
      xdsub(num1, num2, dresult);
      break;
    case OP_MUL:
      xdmul(num1, num2, dresult);
      break;
    case OP_DIV:
      _ddiv(num1, num2, dresult);
      break;
    case OP_CMP:
      return xdcmp(num1, num2);
    case OP_XTOX:
      xdpow(num1, num2, dresult);
      break;
    default:
      goto err_exit;
    }
    return getcon(dresult, DT_DBLE);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    qnum1[0] = CONVAL1G(conval1);
    qnum1[1] = CONVAL2G(conval1);
    qnum1[2] = CONVAL3G(conval1);
    qnum1[3] = CONVAL4G(conval1);
    qnum2[0] = CONVAL1G(conval2);
    qnum2[1] = CONVAL2G(conval2);
    qnum2[2] = CONVAL3G(conval2);
    qnum2[3] = CONVAL4G(conval2);
    switch (opr) {
    case OP_ADD:
      xqadd(qnum1, qnum2, qresult);
      break;
    case OP_SUB:
      xqsub(qnum1, qnum2, qresult);
      break;
    case OP_MUL:
      xqmul(qnum1, qnum2, qresult);
      break;
    case OP_DIV:
      xqdiv(qnum1, qnum2, qresult);
      break;
    case OP_CMP:
      return xqcmp(qnum1, qnum2);
    case OP_XTOX:
      xqpow(qnum1, qnum2, qresult);
      break;
    default:
      goto err_exit;
    }
    return getcon(qresult, DT_QUAD);
#endif

  case TY_CMPLX:
    real1 = CONVAL1G(conval1);
    imag1 = CONVAL2G(conval1);
    real2 = CONVAL1G(conval2);
    imag2 = CONVAL2G(conval2);
    switch (opr) {
    case OP_ADD:
      xfadd(real1, real2, &realrs);
      xfadd(imag1, imag2, &imagrs);
      break;
    case OP_SUB:
      xfsub(real1, real2, &realrs);
      xfsub(imag1, imag2, &imagrs);
      break;
    case OP_MUL:
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      xfmul(real1, real2, &temp1);
      xfmul(imag1, imag2, &temp);
      xfsub(temp1, temp, &realrs);
      xfmul(real1, imag2, &temp1);
      xfmul(real2, imag1, &temp);
      xfadd(temp1, temp, &imagrs);
      break;
    case OP_DIV:
      /*
       *  realrs = real2;
       *  if (realrs < 0)
       *      realrs = -realrs;
       *  imagrs = imag2;
       *  if (imagrs < 0)
       *      imagrs = -imagrs;
       */
      if (xfcmp(real2, CONVAL2G(stb.flt0)) < 0)
        xfsub(CONVAL2G(stb.flt0), real2, &realrs);
      else
        realrs = real2;

      if (xfcmp(imag2, CONVAL2G(stb.flt0)) < 0)
        xfsub(CONVAL2G(stb.flt0), imag2, &imagrs);
      else
        imagrs = imag2;

      /* avoid overflow */

      if (xfcmp(realrs, imagrs) <= 0) {
        /*
         *  if (realrs <= imagrs) {
         *      temp = real2 / imag2;
         *      temp1 = 1.0f / (imag2 * (1 + temp * temp));
         *      realrs = (real1 * temp + imag1) * temp1;
         *      imagrs = (imag1 * temp - real1) * temp1;
         *  }
         */
        temp = _fdiv(real2, imag2);

        xfmul(temp, temp, &temp1);
        xfadd(CONVAL2G(stb.flt1), temp1, &temp1);
        xfmul(imag2, temp1, &temp1);
        temp1 = _fdiv(CONVAL2G(stb.flt1), temp1);

        xfmul(real1, temp, &realrs);
        xfadd(realrs, imag1, &realrs);
        xfmul(realrs, temp1, &realrs);

        xfmul(imag1, temp, &imagrs);
        xfsub(imagrs, real1, &imagrs);
        xfmul(imagrs, temp1, &imagrs);
      } else {
        /*
         *  else {
         *      temp = imag2 / real2;
         *      temp1 = 1.0f / (real2 * (1 + temp * temp));
         *      realrs = (real1 + imag1 * temp) * temp1;
         *      imagrs = (imag1 - real1 * temp) * temp1;
         *  }
         */
        temp = _fdiv(imag2, real2);

        xfmul(temp, temp, &temp1);
        xfadd(CONVAL2G(stb.flt1), temp1, &temp1);
        xfmul(real2, temp1, &temp1);
        temp1 = _fdiv(CONVAL2G(stb.flt1), temp1);

        xfmul(imag1, temp, &realrs);
        xfadd(real1, realrs, &realrs);
        xfmul(realrs, temp1, &realrs);

        xfmul(real1, temp, &imagrs);
        xfsub(imag1, imagrs, &imagrs);
        xfmul(imagrs, temp1, &imagrs);
      }
      break;
    case OP_CMP:
      /*
       * for complex, only EQ and NE comparisons are allowed, so return
       * 0 if the two constants are the same, else 1:
       */
      return (conval1 != conval2);
    case OP_XTOX:
      xcfpow(real1, imag1, real2, imag2, &realrs, &imagrs);
      break;
    default:
      goto err_exit;
    }
    num1[0] = realrs;
    num1[1] = imagrs;
    return getcon(num1, DT_CMPLX);

  case TY_DCMPLX:
    dreal1[0] = CONVAL1G(CONVAL1G(conval1));
    dreal1[1] = CONVAL2G(CONVAL1G(conval1));
    dimag1[0] = CONVAL1G(CONVAL2G(conval1));
    dimag1[1] = CONVAL2G(CONVAL2G(conval1));
    dreal2[0] = CONVAL1G(CONVAL1G(conval2));
    dreal2[1] = CONVAL2G(CONVAL1G(conval2));
    dimag2[0] = CONVAL1G(CONVAL2G(conval2));
    dimag2[1] = CONVAL2G(CONVAL2G(conval2));
    switch (opr) {
    case OP_ADD:
      xdadd(dreal1, dreal2, drealrs);
      xdadd(dimag1, dimag2, dimagrs);
      break;
    case OP_SUB:
      xdsub(dreal1, dreal2, drealrs);
      xdsub(dimag1, dimag2, dimagrs);
      break;
    case OP_MUL:
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      xdmul(dreal1, dreal2, dtemp1);
      xdmul(dimag1, dimag2, dtemp);
      xdsub(dtemp1, dtemp, drealrs);
      xdmul(dreal1, dimag2, dtemp1);
      xdmul(dreal2, dimag1, dtemp);
      xdadd(dtemp1, dtemp, dimagrs);
      break;
    case OP_DIV:
      dtemp2[0] = CONVAL1G(stb.dbl0);
      dtemp2[1] = CONVAL2G(stb.dbl0);
      /*  drealrs = dreal2;
       *  if (drealrs < 0)
       *      drealrs = -drealrs;
       *  dimagrs = dimag2;
       *  if (dimagrs < 0)
       *      dimagrs = -dimagrs;
       */
      if (xdcmp(dreal2, dtemp2) < 0)
        xdsub(dtemp2, dreal2, drealrs);
      else {
        drealrs[0] = dreal2[0];
        drealrs[1] = dreal2[1];
      }
      if (xdcmp(dimag2, dtemp2) < 0)
        xdsub(dtemp2, dimag2, dimagrs);
      else {
        dimagrs[0] = dimag2[0];
        dimagrs[1] = dimag2[1];
      }

      /* avoid overflow */

      dtemp2[0] = CONVAL1G(stb.dbl1);
      dtemp2[1] = CONVAL2G(stb.dbl1);
      if (xdcmp(drealrs, dimagrs) <= 0) {
        /*  if (drealrs <= dimagrs) {
         *     dtemp = dreal2 / dimag2;
         *     dtemp1 = 1.0 / (dimag2 * (1 + dtemp * dtemp));
         *     drealrs = (dreal1 * dtemp + dimag1) * dtemp1;
         *     dimagrs = (dimag1 * dtemp - dreal1) * dtemp1;
         *  }
         */
        _ddiv(dreal2, dimag2, dtemp);

        xdmul(dtemp, dtemp, dtemp1);
        xdadd(dtemp2, dtemp1, dtemp1);
        xdmul(dimag2, dtemp1, dtemp1);
        _ddiv(dtemp2, dtemp1, dtemp1);

        xdmul(dreal1, dtemp, drealrs);
        xdadd(drealrs, dimag1, drealrs);
        xdmul(drealrs, dtemp1, drealrs);

        xdmul(dimag1, dtemp, dimagrs);
        xdsub(dimagrs, dreal1, dimagrs);
        xdmul(dimagrs, dtemp1, dimagrs);
      } else {
        /*  else {
         *  	dtemp = dimag2 / dreal2;
         *  	dtemp1 = 1.0 / (dreal2 * (1 + dtemp * dtemp));
         *  	drealrs = (dreal1 + dimag1 * dtemp) * dtemp1;
         *  	dimagrs = (dimag1 - dreal1 * dtemp) * dtemp1;
         *  }
         */
        _ddiv(dimag2, dreal2, dtemp);

        xdmul(dtemp, dtemp, dtemp1);
        xdadd(dtemp2, dtemp1, dtemp1);
        xdmul(dreal2, dtemp1, dtemp1);
        _ddiv(dtemp2, dtemp1, dtemp1);

        xdmul(dimag1, dtemp, drealrs);
        xdadd(dreal1, drealrs, drealrs);
        xdmul(drealrs, dtemp1, drealrs);

        xdmul(dreal1, dtemp, dimagrs);
        xdsub(dimag1, dimagrs, dimagrs);
        xdmul(dimagrs, dtemp1, dimagrs);
      }
      break;
    case OP_CMP:
      /*
       * for complex, only EQ and NE comparisons are allowed, so return
       * 0 if the two constants are the same, else 1:
       */
      return (conval1 != conval2);
    case OP_XTOX:
      xcdpow(dreal1, dimag1, dreal2, dimag2, drealrs, dimagrs);
      break;
    default:
      goto err_exit;
    }

    num1[0] = getcon(drealrs, DT_DBLE);
    num1[1] = getcon(dimagrs, DT_DBLE);
    return getcon(num1, DT_DCMPLX);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    qreal1[0] = CONVAL1G(CONVAL1G(conval1));
    qreal1[1] = CONVAL2G(CONVAL1G(conval1));
    qreal1[2] = CONVAL3G(CONVAL1G(conval1));
    qreal1[3] = CONVAL4G(CONVAL1G(conval1));
    qimag1[0] = CONVAL1G(CONVAL2G(conval1));
    qimag1[1] = CONVAL2G(CONVAL2G(conval1));
    qimag1[2] = CONVAL3G(CONVAL2G(conval1));
    qimag1[3] = CONVAL4G(CONVAL2G(conval1));
    qreal2[0] = CONVAL1G(CONVAL1G(conval2));
    qreal2[1] = CONVAL2G(CONVAL1G(conval2));
    qreal2[2] = CONVAL3G(CONVAL1G(conval2));
    qreal2[3] = CONVAL4G(CONVAL1G(conval2));
    qimag2[0] = CONVAL1G(CONVAL2G(conval2));
    qimag2[1] = CONVAL2G(CONVAL2G(conval2));
    qimag2[2] = CONVAL3G(CONVAL2G(conval2));
    qimag2[3] = CONVAL4G(CONVAL2G(conval2));
    switch (opr) {
    case OP_ADD:
      xqadd(qreal1, qreal2, qrealrs);
      xqadd(qimag1, qimag2, qimagrs);
      break;
    case OP_SUB:
      xqsub(qreal1, qreal2, qrealrs);
      xqsub(qimag1, qimag2, qimagrs);
      break;
    case OP_MUL:
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      xqmul(qreal1, qreal2, qtemp1);
      xqmul(qimag1, qimag2, qtemp);
      xqsub(qtemp1, qtemp, qrealrs);
      xqmul(qreal1, qimag2, qtemp1);
      xqmul(qreal2, qimag1, qtemp);
      xqadd(qtemp1, qtemp, qimagrs);
      break;
    case OP_DIV:
      qtemp2[0] = CONVAL1G(stb.quad0);
      qtemp2[1] = CONVAL2G(stb.quad0);
      qtemp2[2] = CONVAL3G(stb.quad0);
      qtemp2[3] = CONVAL4G(stb.quad0);
      /*  qrealrs = qreal2;
       *  if (qrealrs < 0)
       *      qrealrs = -qrealrs;
       *  qimagrs = qimag2;
       *  if (qimagrs < 0)
       *      qimagrs = -qimagrs;
       */
      if (xqcmp(qreal2, qtemp2) < 0)
        xqsub(qtemp2, qreal2, qrealrs);
      else {
        qrealrs[0] = qreal2[0];
        qrealrs[1] = qreal2[1];
        qrealrs[2] = qreal2[2];
        qrealrs[3] = qreal2[3];
      }
      if (xqcmp(qimag2, qtemp2) < 0)
        xqsub(qtemp2, qimag2, qimagrs);
      else {
        qimagrs[0] = qimag2[0];
        qimagrs[1] = qimag2[1];
        qimagrs[2] = qimag2[2];
        qimagrs[3] = qimag2[3];
      }

      /* avoid overflow */

      qtemp2[0] = CONVAL1G(stb.quad1);
      qtemp2[1] = CONVAL2G(stb.quad1);
      qtemp2[2] = CONVAL3G(stb.quad1);
      qtemp2[3] = CONVAL4G(stb.quad1);
      if (xqcmp(qrealrs, qimagrs) <= 0) {
        /*  if (qrealrs <= qimagrs) {
         *     qtemp = qreal2 / qimag2;
         *     qtemp1 = 1.0 / (qimag2 * (1 + qtemp * qtemp));
         *     qrealrs = (qreal1 * qtemp + qimag1) * qtemp1;
         *     qimagrs = (qimag1 * qtemp - qreal1) * qtemp1;
         *  }
         */
        xqdiv(qreal2, qimag2, qtemp);

        xqmul(qtemp, qtemp, qtemp1);
        xqadd(qtemp2, qtemp1, qtemp1);
        xqmul(qimag2, qtemp1, qtemp1);
        xqdiv(qtemp2, qtemp1, qtemp1);

        xqmul(qreal1, qtemp, qrealrs);
        xqadd(qrealrs, qimag1, qrealrs);
        xqmul(qrealrs, qtemp1, qrealrs);

        xqmul(qimag1, qtemp, qimagrs);
        xqsub(qimagrs, qreal1, qimagrs);
        xqmul(qimagrs, qtemp1, qimagrs);
      } else {
        /*  else {
         *  	qtemp = qimag2 / qreal2;
         *  	qtemp1 = 1.0 / (qreal2 * (1 + qtemp * qtemp));
         *  	qrealrs = (qreal1 + qimag1 * qtemp) * qtemp1;
         *  	qimagrs = (qimag1 - qreal1 * qtemp) * qtemp1;
         *  }
         */
        xqdiv(qimag2, qreal2, qtemp);

        xqmul(qtemp, qtemp, qtemp1);
        xqadd(qtemp2, qtemp1, qtemp1);
        xqmul(qreal2, qtemp1, qtemp1);
        xqdiv(qtemp2, qtemp1, qtemp1);

        xqmul(qimag1, qtemp, qrealrs);
        xqadd(qreal1, qrealrs, qrealrs);
        xqmul(qrealrs, qtemp1, qrealrs);

        xqmul(qreal1, qtemp, qimagrs);
        xqsub(qimag1, qimagrs, qimagrs);
        xqmul(qimagrs, qtemp1, qimagrs);
      }
      break;
    case OP_CMP:
      /*
       * for complex, only EQ and NE comparisons are allowed, so return
       * 0 if the two constants are the same, else 1:
       */
      return (conval1 != conval2);
    case OP_XTOX:
      xcqpow(qreal1, qimag1, qreal2, qimag2, qrealrs, qimagrs);
      break;
    default:
      goto err_exit;
    }

    num1[0] = getcon(qrealrs, DT_QUAD);
    num1[1] = getcon(qimagrs, DT_QUAD);
    return getcon(num1, DT_QCMPLX);
#endif

  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
    if (opr != OP_CMP) {
      goto err_exit;
    }
    /*
     * opr is assumed to be OP_CMP, only EQ and NE comparisons are
     * allowed so just return 0 if eq, else 1:
     */
    return (conval1 != conval2);
  case TY_NCHAR:
    if (opr != OP_CMP) {
      goto err_exit;
    }
#define KANJI_BLANK 0xA1A1
    {
      int bytes, val1, val2;
      /* following if condition prevent seg fault from following example;
       * logical ::b=char(32,kind=2).eq.char(45,kind=2)
       */
      if (CONVAL1G(conval1) > stb.stg_avail || CONVAL1G(conval2) > stb.stg_avail) {
        interr(
            "init_fold_const: value of kind is not supported in this context",
            dtype, ERR_Severe);
        return (0);
      }

      cvlen1 = DTY(DTYPEG(CONVAL1G(conval1))) + 1;
      cvlen2 = DTY(DTYPEG(CONVAL1G(conval2))) + 1;
      p = stb.n_base + CONVAL1G(CONVAL1G(conval1));
      q = stb.n_base + CONVAL1G(CONVAL1G(conval2));

      while (cvlen1 > 0 && cvlen2 > 0) {
        val1 = kanji_char((unsigned char *)p, cvlen1, &bytes);
        p += bytes, cvlen1 -= bytes;
        val2 = kanji_char((unsigned char *)q, cvlen2, &bytes);
        q += bytes, cvlen2 -= bytes;
        if (val1 != val2)
          return (val1 - val2);
      }

      while (cvlen1 > 0) {
        val1 = kanji_char((unsigned char *)p, cvlen1, &bytes);
        p += bytes, cvlen1 -= bytes;
        if (val1 != KANJI_BLANK)
          return (val1 - KANJI_BLANK);
      }

      while (cvlen2 > 0) {
        val2 = kanji_char((unsigned char *)q, cvlen2, &bytes);
        q += bytes, cvlen2 -= bytes;
        if (val2 != KANJI_BLANK)
          return (KANJI_BLANK - val2);
      }
    }
    return 0;

  case TY_CHAR:
    if (opr != OP_CMP) {
      goto err_exit;
    }
    /* opr is OP_CMP, return -1, 0, or 1:  */
    cvlen1 = DTyCharLength(DTYPEG(conval1));
    cvlen2 = DTyCharLength(DTYPEG(conval2));
    if (cvlen1 == 0 || cvlen2 == 0) {
      return cvlen1 - cvlen2;
    }
    /* change the shorter string to be of same length as the longer: */
    if (cvlen1 < cvlen2) {
      conval1 = cngcon(conval1, DTYPEG(conval1), DTYPEG(conval2));
      cvlen1 = cvlen2;
    } else {
      conval2 = cngcon(conval2, DTYPEG(conval2), DTYPEG(conval1));
    }

    p = stb.n_base + CONVAL1G(conval1);
    q = stb.n_base + CONVAL1G(conval2);
    do {
      if (*p != *q)
        return (*p - *q);
      ++p;
      ++q;
    } while (--cvlen1);
    return 0;
  }

err_exit:
  interr("init_fold_const: bad args", dtype, ERR_Severe);
  return (0);
}

/* Routine init_negate_const stolen from file ast.c in Fortran frontend */
static INT
init_negate_const(INT conval, DTYPE dtype)
{
  SNGL result;
  DBLE drealrs, dimagrs;
#ifdef TARGET_SUPPORTS_QUADFP
  QUAD qrealrs, qimagrs;
#endif
  static INT num[4];

  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
    return (-conval);

  case TY_INT8:
  case TY_LOG8:
    return init_fold_const(OP_SUB, (INT)stb.k0, conval, dtype);

  case TY_REAL:
    xfneg(conval, &result);
    return (result);

  case TY_DBLE:
    num[0] = CONVAL1G(conval);
    num[1] = CONVAL2G(conval);
    xdneg(num, drealrs);
    return getcon(drealrs, DT_DBLE);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    num[0] = CONVAL1G(conval);
    num[1] = CONVAL2G(conval);
    num[2] = CONVAL3G(conval);
    num[3] = CONVAL4G(conval);
    xqneg(num, qrealrs);
    return getcon(qrealrs, DT_QUAD);
#endif

  case TY_CMPLX:
    xfneg(CONVAL1G(conval), &num[0]); /* real part */
    xfneg(CONVAL2G(conval), &num[1]); /* imag part */
    return getcon(num, DT_CMPLX);

  case TY_DCMPLX:
    num[0] = CONVAL1G(CONVAL1G(conval));
    num[1] = CONVAL2G(CONVAL1G(conval));
    xdneg(num, drealrs);
    num[0] = CONVAL1G(CONVAL2G(conval));
    num[1] = CONVAL2G(CONVAL2G(conval));
    xdneg(num, dimagrs);
    num[0] = getcon(drealrs, DT_DBLE);
    num[1] = getcon(dimagrs, DT_DBLE);
    return getcon(num, DT_DCMPLX);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    num[0] = CONVAL1G(CONVAL1G(conval));
    num[1] = CONVAL2G(CONVAL1G(conval));
    num[2] = CONVAL3G(CONVAL1G(conval));
    num[3] = CONVAL4G(CONVAL1G(conval));
    xqneg(num, qrealrs);
    num[0] = CONVAL1G(CONVAL2G(conval));
    num[1] = CONVAL2G(CONVAL2G(conval));
    num[2] = CONVAL3G(CONVAL2G(conval));
    num[3] = CONVAL4G(CONVAL2G(conval));
    xqneg(num, qimagrs);
    num[0] = getcon(qrealrs, DT_QUAD);
    num[1] = getcon(qimagrs, DT_QUAD);
    return getcon(num, DT_QCMPLX);
#endif

  default:
    interr("init_negate_const: bad dtype", dtype, ERR_Severe);
    return (0);
  }
}

static struct {
  CONST *root;
  CONST *roottail;
  CONST *arrbase;
  int ndims;
  struct {
    DTYPE dtype;
    ISZ_T idx;
    CONST *subscr_base;
    ISZ_T lowb;
    ISZ_T upb;
    ISZ_T stride;
  } sub[7];
  struct {
    ISZ_T lowb;
    ISZ_T upb;
    ISZ_T mplyr;
  } dim[7];
} sb;

static ISZ_T
eval_sub_index(int dim)
{
  int repeatc;
  ISZ_T o_lowb, elem_offset;
  CONST *subscr_base;
  ADSC *adsc = AD_DPTR(sb.sub[dim].dtype);
  o_lowb = ad_val_of(AD_LWBD(adsc, 0));
  subscr_base = sb.sub[dim].subscr_base;

  elem_offset = (sb.sub[dim].idx - o_lowb);
  while (elem_offset && subscr_base) {
    if (subscr_base->repeatc > 1) {
      repeatc = subscr_base->repeatc;
      while (repeatc > 0 && elem_offset) {
        --repeatc;
        --elem_offset;
      }
    } else {
      subscr_base = subscr_base->next;
      --elem_offset;
    }
  }
  return get_ival(subscr_base->dtype, subscr_base->u1.conval);
}

static int
eval_sb(int d)
{
  int i;
  ISZ_T sub_idx;
  ISZ_T elem_offset;
  ISZ_T repeat;
  int t_ub = 0;
  CONST *v;
  CONST *c;
  CONST tmp;

#define TRACE_EVAL_SB 0
  if (d == 0) {
#if TRACE_EVAL_SB
    printf("-----\n");
#endif
    sb.sub[0].idx = sb.sub[0].lowb;
    /* Need to also handle negative stride of subscript triplets */
    if (sb.sub[0].stride > 0) {
      t_ub = 1;
    }
    while ((t_ub ? sb.sub[0].idx <= sb.sub[0].upb
                 : sb.sub[0].idx >= sb.sub[0].upb)) {
      /* compute element offset */
      elem_offset = 0;
      for (i = 0; i < sb.ndims; i++) {
        sub_idx = sb.sub[i].idx;
        if (sb.sub[i].subscr_base) {
          sub_idx = eval_sub_index(i);
        }
        elem_offset += (sub_idx - sb.dim[i].lowb) * sb.dim[i].mplyr;
#if TRACE_EVAL_SB
        printf("%3d ", sub_idx);
#endif
      }
#if TRACE_EVAL_SB
      printf(" elem_offset - %ld\n", elem_offset);
#endif
      /* get initialization value at element offset */
      v = sb.arrbase;
      while (v && elem_offset) {
        repeat = v->repeatc;
        if (repeat > 1) {
          while (repeat > 0 && elem_offset) {
            --elem_offset;
            --repeat;
          }
        } else {
          v = v->next;
          --elem_offset;
        }
      }
      if (v == NULL) {
        interr("initialization expression: invalid array subscripts\n",
               elem_offset, ERR_Severe);
        return 1;
      }
      /*
       * evaluate initialization value and add (repeat copies) to
       * initialization list
       */
      tmp = *v;
      tmp.next = 0;
      tmp.repeatc = 1;
      c = eval_init_expr_item(clone_init_const(&tmp, true));
      c->next = NULL;

      add_to_list(c, &sb.root, &sb.roottail);
      sb.sub[0].idx += sb.sub[0].stride;
    }
#if TRACE_EVAL_SB
    printf("-----\n");
#endif
    return 0;
  }
  if (sb.sub[d].stride > 0) {
    for (sb.sub[d].idx = sb.sub[d].lowb; sb.sub[d].idx <= sb.sub[d].upb;
         sb.sub[d].idx += sb.sub[d].stride) {
      if (eval_sb(d - 1))
        return 1;
    }
  } else {
    for (sb.sub[d].idx = sb.sub[d].lowb; sb.sub[d].idx >= sb.sub[d].upb;
         sb.sub[d].idx += sb.sub[d].stride) {
      if (eval_sb(d - 1))
        return 1;
    }
  }
  return 0;
}

static CONST *
eval_const_array_triple_section(CONST *curr_e)
{
  DTYPE dtype;
  CONST *c, *lop, *rop;
  CONST *v;
  int ndims = 0;

  sb.root = sb.roottail = NULL;
  c = curr_e;
  do {
    rop = c->u1.expr.rop;
    lop = c->u1.expr.lop;
    sb.sub[ndims].subscr_base = 0;
    sb.sub[ndims].dtype = DT_NONE;
    /* Due to how we read in EXPR in upper.c if the lop is null the rop
     * will be put on lop instead. */
    if (rop) {
      dtype = rop->dtype;
      sb.sub[ndims].dtype = lop->dtype;
    }
    if (rop == NULL) {
      rop = lop;
      dtype = rop->dtype;
    } else if (lop) {
      CONST *t = eval_init_expr(lop);
      if (t->id == AC_ACONST)
        sb.sub[ndims].subscr_base = t->subc;
      else
        sb.sub[ndims].subscr_base = t;
    }
    /* Need to keep dtype of the original array to get actual lower/upper
     * bound when we evaluate subscript later on.
     */

    if (rop == 0) {
      interr("initialization expression: missing array section lb\n", 0, ERR_Severe);
      return CONST_ERR(dtype);
    }
    v = eval_init_expr(rop);
    if (!v || v->id != AC_CONST) {
      interr("initialization expression: non-constant lb\n", 0, ERR_Severe);
      return CONST_ERR(dtype);
    }
    sb.sub[ndims].lowb = get_ival(v->dtype, v->u1.conval);

    if ((rop = rop->next) == 0) {
      interr("initialization expression: missing array section ub\n", 0, ERR_Severe);
      return CONST_ERR(dtype);
    }
    v = eval_init_expr(rop);
    if (!v || v->id != AC_CONST) {
      interr("initialization expression: non-constant ub\n", 0, ERR_Severe);
      return CONST_ERR(dtype);
    }
    sb.sub[ndims].upb = get_ival(v->dtype, v->u1.conval);

    if ((rop = rop->next) == 0) {
      interr("initialization expression: missing array section stride\n", 0, ERR_Severe);
      return CONST_ERR(dtype);
    }
    v = eval_init_expr(rop);
    if (!v || v->id != AC_CONST) {
      interr("initialization expression: non-constant stride\n", 0, ERR_Severe);
      return CONST_ERR(dtype);
    }
    sb.sub[ndims].stride = get_ival(v->dtype, v->u1.conval);

    if (++ndims >= 7) {
      interr("initialization expression: too many dimensions\n", 0, ERR_Severe);
      return CONST_ERR(dtype);
    }
    c = c->next;
  } while (c);

  sb.ndims = ndims;
  return sb.root;
}

static CONST *
eval_const_array_section(CONST *lop, DTYPE ldtype, DTYPE dtype)
{
  ADSC *adsc = AD_DPTR(ldtype);
  int ndims = 0;
  int i;

  sb.root = sb.roottail = NULL;
  if (lop->id == AC_ACONST) {
    sb.arrbase = eval_array_constructor(lop);
  } else {
    sb.arrbase = lop;
  }

  if (sb.ndims != AD_NUMDIM(adsc)) {
    interr("initialization expression: subscript/dimension mis-match\n", ldtype,
           ERR_Severe);
    return CONST_ERR(dtype);
  }
  ndims = AD_NUMDIM(adsc);
  for (i = 0; i < ndims; i++) {
    sb.dim[i].lowb = ad_val_of(AD_LWBD(adsc, i));
    sb.dim[i].upb = ad_val_of(AD_UPBD(adsc, i));
    sb.dim[i].mplyr = ad_val_of(AD_MLPYR(adsc, i));
  }

  sb.ndims = ndims;
  if (eval_sb(ndims - 1))
    return CONST_ERR(dtype);

  return sb.root;
}

#define ABS(x) (((x) > 0) ? (x) : (-(x)))
/** \brief Evaluate compile-time constant produced by ISFHT intrinsic
 *
 * ISHFT(I, SHIFT) shifts value in I by SHIFT bits to the left (if SHIFT is
 * negative, it shifts by -SHIFT to the right).
 *
 * For 64-bit values we need to extract the arguments from the symbol table and
 * write the result back to it.
 *
 * \param arg initilization expression
 * \param dtype expected result type
 */
INLINE static CONST *
eval_ishft(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  CONST *arg2 = eval_init_expr_item(arg->next);
  ISZ_T val, shftval;
  INT ival[2];

  /* Get shift value
   *
   * 32-bit values get stored in the conval field, while larger values need to
   * be looked up in the symbol table.
   */
  if (size_of(arg2->dtype) > 4) {
    ival[0] = CONVAL1G(arg2->u1.conval);
    ival[1] = CONVAL2G(arg2->u1.conval);
    INT64_2_ISZ(ival, shftval);
  } else {
    shftval = arg2->u1.conval;
  }

  /* Check whether shift value is within the size of the argument */
  if (ABS(shftval) > dtypeinfo[wrkarg->dtype].bits) {
    error(S_0282_ISHFT_shift_is_greater_than_the_bit_size_of_the_value_argument,
          ERR_Severe, gbl.lineno, NULL, NULL);
    return CONST_ERR(dtype);
  }

  for (; wrkarg; wrkarg = wrkarg->next) {
    /* Get the first argument to ishft
     *
     * 32-bit values get stored in the conval field, while larger values need
     * to be looked up in the symbol table.
     */
    if (size_of(wrkarg->dtype) > 4) {
      ival[0] = CONVAL1G(wrkarg->u1.conval);
      ival[1] = CONVAL2G(wrkarg->u1.conval);
      INT64_2_ISZ(ival, val);
    } else {
      val = wrkarg->u1.conval;
    }

    /* Shift */
    if (shftval < 0) {
      val >>= -shftval;
    }
    if (shftval > 0) {
      val <<= shftval;
    }

    /* Write back
     *
     * 32-bit values get stored in the conval field, while larger values need
     * to be put into the symbol table.
     */
    if (size_of(wrkarg->dtype) > 4) {
      ISZ_2_INT64(val, ival);
      wrkarg->u1.conval = getcon(ival, rslt->dtype);
    } else {
      wrkarg->u1.conval = val;
    }
  }

  return rslt;
}

#define INTINTRIN2(iname, ent, op)                                  \
  static CONST *ent(CONST *arg, DTYPE dtype)                        \
  {                                                                 \
    CONST *arg1 = eval_init_expr_item(arg);                         \
    CONST *arg2 = eval_init_expr_item(arg->next);                   \
    CONST *rslt = clone_init_const_list(arg1, true);                \
    arg1 = rslt->id == AC_ACONST ? rslt->subc : rslt;               \
    arg2 = arg2->id == AC_ACONST ? arg2->subc : arg2;               \
    for (; arg1; arg1 = arg1->next, arg2 = arg2->next) {            \
      int con1 = arg1->u1.conval;                                   \
      int con2 = arg2->u1.conval;                                   \
      int num1[2], num2[2], res[2], conval;                         \
      if (DT_ISWORD(arg1->dtype)) {                                 \
        num1[0] = 0, num1[1] = con1;                                \
      } else {                                                      \
        num1[0] = CONVAL1G(con1), num1[1] = CONVAL2G(con1);         \
      }                                                             \
      if (DT_ISWORD(arg2->dtype)) {                                 \
        num2[0] = 0, num2[1] = con2;                                \
      } else {                                                      \
        num2[0] = CONVAL1G(con2), num2[1] = CONVAL2G(con2);         \
      }                                                             \
      res[0] = num1[0] op num2[0];                                  \
      res[1] = num1[1] op num2[1];                                  \
      conval = DT_ISWORD(dtype) ? res[1] : getcon(res, DT_INT8);    \
      arg1->u1.conval = conval;                                     \
      arg1->dtype = dtype;                                          \
      arg1->id = AC_CONST;                                          \
      arg1->repeatc = 1;                                            \
    }                                                               \
    return rslt;                                                    \
  }

INTINTRIN2("iand", eval_iand, &)
INTINTRIN2("ior", eval_ior, |)
INTINTRIN2("ieor", eval_ieor, ^)

static CONST *
eval_ichar(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr(arg);
  CONST *wrkarg;
  int srcdty;
  DTYPE rsltdtype = DDTG(dtype);
  int clen;
  int c;
  int dum;

  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  srcdty = DTY(wrkarg->dtype);
  for (; wrkarg; wrkarg = wrkarg->next) {
    if (srcdty == TY_NCHAR) {
      c = CONVAL1G(wrkarg->u1.conval);
      clen = size_of(DTYPEG(c));
      c = kanji_char((unsigned char *)stb.n_base + CONVAL1G(c), clen, &dum);
    } else {
      c = stb.n_base[CONVAL1G(wrkarg->u1.conval)] & 0xff;
    }
    wrkarg->u1.conval = cngcon(c, DT_WORD, rsltdtype);
    wrkarg->dtype = rsltdtype;
  }

  rslt->dtype = dtype;
  return rslt;
}

static CONST *
eval_char(CONST *arg, DTYPE dtype)
{
  DTYPE rsltdtype = DDTG(dtype);
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg = rslt->id == AC_ACONST ? rslt->subc : rslt;

  for (; wrkarg; wrkarg = wrkarg->next) {
    if (DT_ISWORD(wrkarg->dtype)) {
      wrkarg->u1.conval = cngcon(wrkarg->u1.conval, DT_WORD, rsltdtype);
    } else {
      wrkarg->u1.conval = cngcon(wrkarg->u1.conval, DT_DWORD, rsltdtype);
    }
    wrkarg->dtype = rsltdtype;
  }
  rslt->dtype = dtype;
  return rslt;
}

INLINE static CONST *
eval_int(CONST *arg, DTYPE dtype)
{
  INT result;
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg = rslt->id == AC_ACONST ? rslt->subc : rslt;

  for (; wrkarg; wrkarg = wrkarg->next) {
    result = cngcon(wrkarg->u1.conval, wrkarg->dtype, DDTG(dtype));

    wrkarg->id = AC_CONST;
    wrkarg->dtype = DDTG(dtype);
    wrkarg->repeatc = 1;
    wrkarg->u1.conval = result;
  }
  return rslt;
}

static CONST *
eval_null(CONST *arg, DTYPE dtype)
{
  CONST c = {0, NULL, NULL, 0, SPTR_NULL, SPTR_NULL, DT_NONE, 0, {0}};
  CONST *p = clone_init_const(&c, true);
  p->id = AC_CONST;
  p->repeatc = 1;
  p->dtype = DDTG(dtype);
  p->u1.conval = 0;
  return p;
}

static CONST *
eval_fltconvert(CONST *arg, DTYPE dtype)
{
  INT result;
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg = rslt->id == AC_ACONST ? rslt->subc : rslt;

  for (; wrkarg; wrkarg = wrkarg->next) {
    result = cngcon(wrkarg->u1.conval, wrkarg->dtype, DDTG(dtype));

    wrkarg->id = AC_CONST;
    wrkarg->dtype = DDTG(dtype);
    wrkarg->repeatc = 1;
    wrkarg->u1.conval = result;
  }
  return rslt;
}

#define GET_DBLE(x, y) \
  x[0] = CONVAL1G(y);  \
  x[1] = CONVAL2G(y)
#define GET_QUAD(x, y) \
  x[0] = CONVAL1G(y);  \
  x[1] = CONVAL2G(y);  \
  x[2] = CONVAL3G(y);  \
  x[3] = CONVAL4G(y);
#define GETVALI64(x, b) \
  x[0] = CONVAL1G(b);   \
  x[1] = CONVAL2G(b);

static CONST *
eval_abs(CONST *arg, DTYPE dtype)
{
  CONST *rslt;
  CONST *wrkarg;
  INT con1, res[4], num1[4], num2[4];
  DTYPE rsltdtype = dtype;

  rslt = eval_init_expr_item(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    switch (DTY(wrkarg->dtype)) {
    case TY_SINT:
    case TY_BINT:
    case TY_INT:
      con1 = wrkarg->u1.conval;
      if (con1 < 0)
        con1 = -(con1);
      break;
    case TY_INT8:
      con1 = wrkarg->u1.conval;
      GETVALI64(num1, con1);
      GETVALI64(num2, stb.k0);
      if (cmp64(num1, num2) == -1) {
        neg64(num1, res);
        con1 = getcon(res, DT_INT8);
      }
      break;
    case TY_REAL:
      res[0] = 0;
      con1 = wrkarg->u1.conval;
      xfabsv(con1, &res[1]);
      con1 = res[1];
      break;
    case TY_DBLE:
      con1 = wrkarg->u1.conval;
      GET_DBLE(num1, con1);
      xdabsv(num1, res);
      con1 = getcon(res, dtype);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      con1 = wrkarg->u1.conval;
      GET_QUAD(num1, con1);
      xqabsv(num1, res);
      con1 = getcon(res, dtype);
      break;
#endif
    case TY_CMPLX:
      con1 = wrkarg->u1.conval;
      num1[0] = CONVAL1G(con1);
      num1[1] = CONVAL2G(con1);
      xfmul(num1[0], num1[0], &num2[0]);
      xfmul(num1[1], num1[1], &num2[1]);
      xfadd(num2[0], num2[1], &num2[2]);
      xfsqrt(num2[2], &con1);
      wrkarg->dtype = DT_REAL;
      dtype = rsltdtype = DT_REAL;
      break;
    case TY_DCMPLX:
      con1 = wrkarg->u1.conval;
      wrkarg->dtype = DT_DBLE;
      dtype = rsltdtype = DT_DBLE;

      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      con1 = wrkarg->u1.conval;
      GET_QUAD(num1, CONVAL1G(con1));
      GET_QUAD(num2, CONVAL2G(con1));
      xqmul(num1, num1, num1);
      xqmul(num2, num2, num2);
      xqadd(num1, num2, num2);
      xqsqrt(num2, num1);
      con1 = getcon(num1, dtype);
      wrkarg->dtype = DT_QUAD;
      dtype = rsltdtype = DT_QUAD;
      break;
#endif
    default:
      con1 = wrkarg->u1.conval;
      break;
    }

    wrkarg->u1.conval = cngcon(con1, wrkarg->dtype, rsltdtype);
    wrkarg->dtype = dtype;
  }
  return rslt;
}

static CONST *
eval_min(CONST *arg, DTYPE dtype)
{
  CONST **arglist;
  CONST *rslt;
  CONST *wrkarg1;
  CONST *wrkarg2;
  CONST *c, *head;
  CONST *root = NULL;
  CONST *roottail = NULL;
  int repeatc1, repeatc2;
  int nargs;
  int nelems = 0;
  int i, j;

  rslt = (CONST*)getitem(4, sizeof(CONST));
  BZERO(rslt, CONST, 1);
  rslt->dtype = arg->dtype;

  for (wrkarg1 = arg, nargs = 0; wrkarg1; wrkarg1 = wrkarg1->next, nargs++)
    ;
  NEW(arglist, CONST *, nargs);
  for (i = 0, wrkarg1 = arg; i < nargs; i++, wrkarg1 = wrkarg1->next) {
    head = arglist[i] = eval_init_expr(wrkarg1);
    if (DTY(head->dtype) == TY_ARRAY) {
      int num;
      num = ad_val_of(AD_NUMELM(AD_DPTR(head->dtype)));
      if (nelems == 0)
        nelems = num;
      else if (nelems != num)
        ; /* error */

      rslt->id = AC_ACONST;
      rslt->dtype = head->dtype;
    }
  }
  if (nelems == 0) {
    nelems = 1;
    c = rslt;
    c->id = AC_CONST;
    c->repeatc = 0;
    c->next = NULL;
    add_to_list(c, &root, &roottail);
  } else {
    for (i = 0; i < nelems; i++) {
      c = (CONST*)getitem(4, sizeof(CONST));
      BZERO(c, CONST, 1);
      c->id = AC_CONST;
      c->repeatc = 1;
      add_to_list(c, &root, &roottail);
    }
    rslt->subc = root;
    rslt->repeatc = 0;
  }

  wrkarg1 = arglist[0];
  for (j = 1; j < nargs; j++) {
    wrkarg2 = arglist[j];
    if (wrkarg2->id == AC_ACONST) {
      wrkarg2 = wrkarg2->subc;
      repeatc2 = wrkarg2->repeatc;
    } else {
      repeatc2 = nelems;
    }
    if (wrkarg1->id == AC_ACONST) {
      wrkarg1 = wrkarg1->subc;
      repeatc1 = wrkarg1->repeatc;
    } else {
      repeatc1 = nelems;
    }

    c = root;
    for (i = 0; i < nelems; i++) {
      if (wrkarg1 != root) {
        c->u1 = wrkarg1->u1;
        c->dtype = wrkarg1->dtype;
      }
      switch (DTY(dtype)) {
      default:
        break;
      case TY_INT:
        if (wrkarg2->u1.conval < wrkarg1->u1.conval) {
          c->u1 = wrkarg2->u1;
        }
        break;
      case TY_CHAR:
        if (strcmp(stb.n_base + CONVAL1G(wrkarg2->u1.conval),
                   stb.n_base + CONVAL1G(wrkarg1->u1.conval)) < 0) {
          c->u1 = wrkarg2->u1;
          c->dtype = wrkarg2->dtype;
        }
        break;
      case TY_REAL:
        if (xfcmp(wrkarg2->u1.conval, wrkarg1->u1.conval) < 0) {
          c->u1 = wrkarg2->u1;
          c->dtype = wrkarg2->dtype;
        }
        break;
      case TY_INT8:
      case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
#endif
        if (init_fold_const(OP_CMP, wrkarg2->u1.conval, wrkarg1->u1.conval,
                            dtype) < 0) {
          c->u1 = wrkarg2->u1;
          c->dtype = wrkarg2->dtype;
        }
        break;
      }
      c = c->next;
      if (root == wrkarg1) {
        wrkarg1 = c;
        repeatc1 = 1;
      } else if (--repeatc1 <= 0) {
        wrkarg1 = wrkarg1->next;
        if (wrkarg1)
          repeatc1 = wrkarg1->repeatc;
      }
      if (--repeatc2 <= 0) {
        wrkarg2 = wrkarg2->next;
        if (wrkarg2) {
          repeatc2 = wrkarg2->repeatc;
        } 
      }
      
    }
    wrkarg1 = c = root;
  }
  FREE(arglist);

  return rslt;
}

static CONST *
eval_max(CONST *arg, DTYPE dtype)
{
  CONST **arglist;
  CONST *rslt;
  CONST *wrkarg1;
  CONST *wrkarg2;
  CONST *c, *head;
  CONST *root = NULL;
  CONST *roottail = NULL;
  int repeatc1, repeatc2;
  int nargs;
  int nelems = 0;
  int i, j;

  rslt = (CONST*)getitem(4, sizeof(CONST));
  BZERO(rslt, CONST, 1);
  rslt->dtype = arg->dtype;

  for (wrkarg1 = arg, nargs = 0; wrkarg1; wrkarg1 = wrkarg1->next, nargs++)
    ;
  NEW(arglist, CONST *, nargs);
  for (i = 0, wrkarg1 = arg; i < nargs; i++, wrkarg1 = wrkarg1->next) {
    head = arglist[i] = eval_init_expr(wrkarg1);
    if (DTY(head->dtype) == TY_ARRAY) {
      int num;
      num = ad_val_of(AD_NUMELM(AD_DPTR(head->dtype)));
      if (nelems == 0)
        nelems = num;
      else if (nelems != num)
        ; /* error */

      rslt->id = AC_ACONST;
      rslt->dtype = head->dtype;
    }
  }
  if (nelems == 0) {
    nelems = 1;
    c = rslt;
    c->id = AC_CONST;
    c->repeatc = 0;
    c->next = NULL;
    add_to_list(c, &root, &roottail);
  } else {
    for (i = 0; i < nelems; i++) {
      c = (CONST*)getitem(4, sizeof(CONST));
      BZERO(c, CONST, 1);
      c->id = AC_CONST;
      c->repeatc = 1;
      add_to_list(c, &root, &roottail);
    }
    rslt->subc = root;
    rslt->repeatc = 0;
  }

  wrkarg1 = arglist[0];
  for (j = 1; j < nargs; j++) {
    wrkarg2 = arglist[j];
    if (wrkarg2->id == AC_ACONST) {
      wrkarg2 = wrkarg2->subc;
      repeatc2 = wrkarg2->repeatc;
    } else {
      repeatc2 = nelems;
    }
    if (wrkarg1->id == AC_ACONST) {
      wrkarg1 = wrkarg1->subc;
      repeatc1 = wrkarg1->repeatc;
    } else {
      repeatc1 = nelems;
    }

    c = root;
    for (i = 0; i < nelems; i++) {
      if (wrkarg1 != root) {
        c->u1 = wrkarg1->u1;
        c->dtype = wrkarg1->dtype;
      }
      switch (DTY(dtype)) {
      default:
        break;
      case TY_CHAR:
        if (strcmp(stb.n_base + CONVAL1G(wrkarg2->u1.conval),
                   stb.n_base + CONVAL1G(wrkarg1->u1.conval)) > 0) {
          c->u1 = wrkarg2->u1;
          c->dtype = wrkarg2->dtype;
        }
        break;
      case TY_INT:
        if (wrkarg2->u1.conval > wrkarg1->u1.conval) {
          c->u1 = wrkarg2->u1;
          c->dtype = wrkarg2->dtype;
        }
        break;
      case TY_REAL:
        if (xfcmp(wrkarg2->u1.conval, wrkarg1->u1.conval) > 0) {
          c->u1 = wrkarg2->u1;
          c->dtype = wrkarg2->dtype;
        }

        break;
      case TY_INT8:
      case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
#endif
        if (init_fold_const(OP_CMP, wrkarg2->u1.conval, wrkarg1->u1.conval,
                            dtype) > 0) {
          c->u1 = wrkarg2->u1;
          c->dtype = wrkarg2->dtype;
        }
        break;
      }
      c = c->next;
      if (root == wrkarg1) {
        wrkarg1 = c;
        repeatc1 = 1;
      } else if (--repeatc1 <= 0) {
        wrkarg1 = wrkarg1->next;
        if(wrkarg1)
          repeatc1 = wrkarg1->repeatc;
      }
      if (--repeatc2 <= 0) {
        wrkarg2 = wrkarg2->next;
        if (wrkarg2)
          repeatc2 = wrkarg2->repeatc;
      }
    }
    wrkarg1 = c = root;
  }
  FREE(arglist);

  return rslt;
}

/* Compare two constant CONSTs. Return x > y or x < y depending on want_max. */
static bool
cmp_acl(DTYPE dtype, CONST *x, CONST *y, bool want_max, bool back)
{
  int cmp;
  switch (DTY(dtype)) {
  case TY_CHAR:
    cmp = strcmp(stb.n_base + CONVAL1G(x->u1.conval),
                 stb.n_base + CONVAL1G(y->u1.conval));
    break;
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    if (x->u1.conval == y->u1.conval) {
      cmp = 0;
    } else if (x->u1.conval > y->u1.conval) {
      cmp = 1;
    } else {
      cmp = -1;
    } 
    break;
  case TY_REAL:
    cmp = xfcmp(x->u1.conval, y->u1.conval);
    break;
  case TY_INT8:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
    cmp = init_fold_const(OP_CMP, x->u1.conval, y->u1.conval, dtype);
    break;
  default:
    interr("cmp_acl: bad dtype", dtype, ERR_Severe);
    return false;
  }
  if (back) {
    return want_max ? cmp >= 0 : cmp <= 0;
  } else {
    return want_max ? cmp > 0 : cmp < 0;
  }
}

/* An index into a Fortran array. ndims is in [1,MAXDIMS], index[] is the
 * index itself, extent[] is the extent in each dimension.
 * index[i] is in [1,extent[i]] for i in 1..ndims
 */
typedef struct {
  unsigned ndims;
  unsigned index[MAXDIMS + 1];
  unsigned extent[MAXDIMS + 1];
} INDEX;

/* Increment an array index starting at the left and carrying to the right. */
static bool
incr_index(INDEX *index)
{
  unsigned d;
  for (d = 1; d <= index->ndims; ++d) {
    if (index->index[d] < index->extent[d]) {
      index->index[d] += 1;
      return true; /* no carry needed */
    }
    index->index[d] = 1;
  }
  return false;
}

static unsigned
get_offset_without_dim(INDEX *index, unsigned dim)
{
  if (dim == 0) {
    return 0;
  } else {
    unsigned result = 0;
    unsigned d;
    for (d = index->ndims; d > 0; --d) {
      if (d != dim) {
        result *= index->extent[d];
        result += index->index[d] - 1;
      }
    }
    return result;
  }
}

static int
_huge(DTYPE dtype)
{
  INT val[4];
  int tmp;

  switch (DTYG(dtype)) {
  case TY_BINT:
    val[0] = 0x7f;
    goto const_int_val;
  case TY_SINT:
    val[0] = 0x7fff;
    goto const_int_val;
  case TY_INT:
    val[0] = 0x7fffffff;
    goto const_int_val;
  case TY_INT8:
    val[0] = 0x7fffffff;
    val[1] = 0xffffffff;
    goto const_int8_val;
  case TY_REAL:
    /* 3.402823466E+38 */
    val[0] = 0x7f7fffff;
    goto const_real_val;
  case TY_DBLE:
    if (XBIT(49, 0x40000)) {               /* C90 */
#define C90_HUGE "0.136343516952426e+2466" /* 0577757777777777777776 */
      atoxd(C90_HUGE, &val[0], strlen(C90_HUGE));
    } else {
      /* 1.79769313486231571E+308 */
      val[0] = 0x7fefffff;
      val[1] = 0xffffffff;
    }
    goto const_dble_val;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    val[0] = 0x7ffeffff;
    val[1] = 0xffffffff;
    val[2] = 0xffffffff;
    val[3] = 0xffffffff;
    goto const_quad_val;
#endif
  default:
    return 0; /* caller must check */
  }

const_int_val:
  return val[0];
const_int8_val:
  tmp = getcon(val, DT_INT8);
  return tmp;
const_real_val:
  return val[0];
const_dble_val:
  tmp = getcon(val, DT_DBLE);
  return tmp;
#ifdef TARGET_SUPPORTS_QUADFP
const_quad_val:
  tmp = getcon(val, DT_QUAD);
  return tmp;
#endif
}

static INT
negate_const_be(INT conval, DTYPE dtype)
{
  SNGL result, realrs, imagrs;
  DBLE dresult, drealrs, dimagrs;
#ifdef TARGET_SUPPORTS_QUADFP
  IEEE128 qresult, qrealrs, qimagrs;
#endif
  static INT num[4];

  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
    return (-conval);

  case TY_INT8:
  case TY_LOG8:
    return init_fold_const(OP_SUB, (INT)stb.k0, conval, dtype);

  case TY_REAL:
    xfneg(conval, &result);
    return (result);

  case TY_DBLE:
    num[0] = CONVAL1G(conval);
    num[1] = CONVAL2G(conval);
    xdneg(num, dresult);
    return getcon(dresult, DT_DBLE);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    num[0] = CONVAL1G(conval);
    num[1] = CONVAL2G(conval);
    num[2] = CONVAL3G(conval);
    num[3] = CONVAL4G(conval);
    xqneg(num, qresult);
    return getcon(qresult, DT_QUAD);
#endif

  case TY_CMPLX:
    xfneg(CONVAL1G(conval), &realrs);
    xfneg(CONVAL2G(conval), &imagrs);
    num[0] = realrs;
    num[1] = imagrs;
    return getcon(num, DT_CMPLX);

  case TY_DCMPLX:
    dresult[0] = CONVAL1G(CONVAL1G(conval));
    dresult[1] = CONVAL2G(CONVAL1G(conval));
    xdneg(dresult, drealrs);
    dresult[0] = CONVAL1G(CONVAL2G(conval));
    dresult[1] = CONVAL2G(CONVAL2G(conval));
    xdneg(dresult, dimagrs);
    num[0] = getcon(drealrs, DT_DBLE);
    num[1] = getcon(dimagrs, DT_DBLE);
    return getcon(num, DT_DCMPLX);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    qresult[0] = CONVAL1G(CONVAL1G(conval));
    qresult[1] = CONVAL2G(CONVAL1G(conval));
    qresult[2] = CONVAL3G(CONVAL1G(conval));
    qresult[3] = CONVAL4G(CONVAL1G(conval));
    xqneg(qresult, qrealrs);
    qresult[0] = CONVAL1G(CONVAL2G(conval));
    qresult[1] = CONVAL2G(CONVAL2G(conval));
    qresult[2] = CONVAL3G(CONVAL2G(conval));
    qresult[3] = CONVAL4G(CONVAL2G(conval));
    xqneg(qresult, qimagrs);
    num[0] = getcon(qrealrs, DT_QUAD);
    num[1] = getcon(qimagrs, DT_QUAD);
    return getcon(num, DT_QCMPLX);
#endif

  default:
    interr("negate_const: bad dtype", dtype, ERR_Severe);
    return (0);
  }
}

int
mk_unop(int optype, int lop, DTYPE dtype)
{
  switch (optype) {
  case OP_ADD:
    return lop;
  case OP_SUB:
    switch (DTY(dtype)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
    case TY_REAL:
    case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
#endif
    case TY_CMPLX:
    case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
#endif
    case TY_INT8:
    case TY_LOG8:
      return negate_const_be(lop, dtype);
    default:
      interr("mk_unop-negate: bad dtype", dtype, ERR_Severe);
      break;
    }
    break;
  default:
    interr("mk_unop-negate: bad op", optype, ERR_Severe);
    break;
  }
  return lop;
}

int
mk_smallest_val(DTYPE dtype)
{
  INT val[4];
  int tmp;

  switch (DTYG(dtype)) {
  case TY_BINT:
    val[0] = ~0x7f;
    if (XBIT(51, 0x1))
      val[0] |= 0x01;
    break;
  case TY_SINT:
    val[0] = ~0x7fff;
    if (XBIT(51, 0x2))
      val[0] |= 0x0001;
    break;
  case TY_INT:
    val[0] = ~0x7fffffff;
    if (XBIT(51, 0x4))
      val[0] |= 0x00000001;
    break;
  case TY_INT8:
    if (XBIT(49, 0x1040000)) {
      /* T3D/T3E or C90 Cray targets - workaround for cray compiler:
       * -9223372036854775808_8 (-huge()-1) is considered to be out of
       * range; just return -huge().
       */
      tmp = _huge(DT_INT8);
      tmp = mk_unop(OP_SUB, tmp, dtype);
      return tmp;
    }
    val[0] = ~0x7fffffff;
    val[1] = 0;
    if (XBIT(51, 0x8))
      val[1] |= 0x00000001;
    tmp = getcon(val, DT_INT8);
    return tmp;
  case TY_REAL:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
    tmp = _huge(dtype);
    tmp = mk_unop(OP_SUB, tmp, dtype);
    return tmp;
  default:
    return 0; /* caller must check */
  }
  return val[0];
}

int
mk_largest_val(DTYPE dtype)
{
  return _huge(dtype);
}

/* Get a CONST representing the smallest/largest value of this type. */
static CONST *
get_minmax_val(DTYPE dtype, bool want_max)
{
  CONST *temp = (CONST *)getitem(4, sizeof(CONST));
  BZERO(temp, CONST, 1);
  temp->next = 0;
  temp->id = AC_CONST;
  temp->dtype = dtype;
  temp->repeatc = 1;

  temp->u1.conval = want_max ? mk_smallest_val(dtype) : mk_largest_val(dtype);
  return eval_init_expr_item(clone_init_const(temp, true));
}

static CONST *
convert_acl_dtype(CONST *head, DTYPE oldtype, DTYPE newtype)
{
  DTYPE dtype;

  CONST *cur_lop;
  if (DTY(oldtype) == TY_STRUCT || DTY(oldtype) == TY_CHAR ||
      DTY(oldtype) == TY_NCHAR || DTY(oldtype) == TY_UNION) {
    return head;
  }
  cur_lop = head;
  dtype = DDTG(newtype);

  /* make sure all are AC_CONST */
  for (cur_lop = head; cur_lop; cur_lop = cur_lop->next) {
    if (cur_lop->id != AC_CONST)
      return head;
  }

  for (cur_lop = head; cur_lop; cur_lop = cur_lop->next) {
    if (cur_lop->dtype != dtype) {
      cur_lop->u1.conval = cngcon(cur_lop->u1.conval, cur_lop->dtype, dtype);
      cur_lop->dtype = dtype;
    }
  }
  return head;
}

static CONST *
do_eval_minval_or_maxval(INDEX *index, DTYPE elem_dt, DTYPE loc_dt,
                         CONST *elems, unsigned dim, CONST *mask, bool back,
                         int intrin)
{
  unsigned ndims = index->ndims;
  unsigned i;
  CONST **vals;
  unsigned *locs;
  unsigned vals_size = 1;
  unsigned locs_size;
  bool want_max = intrin == AC_I_maxloc || intrin == AC_I_maxval;
  bool want_val = intrin == AC_I_minval || intrin == AC_I_maxval;
 
/* vals[vals_size] contains the result for {min,max}val()
 * locs[locs_size] contains the result for {min,max}loc() */
  if (dim == 0) {
    locs_size = ndims;
  } else {
    for (i = 1; i <= ndims; ++i) {
      if (i != dim)
        vals_size *= index->extent[i];
    }
    locs_size = vals_size;
  }

  NEW(vals, CONST *, vals_size);
  for (i = 0; i < vals_size; ++i) {
    vals[i] = get_minmax_val(elem_dt, want_max);
  }

  NEW(locs, unsigned, locs_size);
  BZERO(locs, unsigned, locs_size);

  { /* iterate over elements computing min/max into vals[] and locs[] */
    CONST *elem;
    for (elem = elems; elem != 0; elem = elem->next) {
      if (elem->dtype != elem_dt) {
        elem = convert_acl_dtype(elem, elem->dtype, elem_dt);
      }
      if (mask->u1.conval) {
        CONST *val = eval_init_expr_item(elem);
        unsigned offset = get_offset_without_dim(index, dim);
        CONST *prev_val = vals[offset];
        if (cmp_acl(elem_dt, val, prev_val, want_max, back)) {
          vals[offset] = val;
          if (dim == 0) {
            BCOPY(locs, &index->index[1], int, ndims);
          } else {
            locs[offset] = index->index[dim];
          }
        }
      }
      if (mask->next)
        mask = mask->next;
      incr_index(index);
    }
  }

  { /* build result from vals[] or locs[] */
    CONST *result;
    CONST *subc = NULL; /* elements of result array */
    CONST *roottail = NULL;
    if (!want_val) {
      for (i = 0; i < locs_size; i++) {
        CONST *elem = (CONST *)getitem(4, sizeof(CONST));
        BZERO(elem, CONST, 1);
        elem->id = AC_CONST;
        elem->dtype = loc_dt;
        elem->u1.conval = locs[i];
        elem->repeatc = 1;
        add_to_list(elem, &subc, &roottail);
      }
    } else if (dim > 0) {
      for (i = 0; i < vals_size; i++) {
        add_to_list(vals[i], &subc, &roottail);
      }
    } else {
      return vals[0]; /* minval/maxval with no dim has scalar result */
    }

    result = (CONST*)getitem(4, sizeof(CONST));;
    BZERO(result, CONST, 1);
    result->id = AC_ACONST;
    result->subc = subc;
    return result;
  }
}

static CONST *
eval_scale(CONST *arg, int type)
{
  CONST *rslt;
  CONST *arg2;
  INT i, conval1, conval2, conval;
  DBLINT64 inum1, inum2;
  INT e;
  DBLE dconval;
#ifdef TARGET_SUPPORTS_QUADFP
  INT qnum1[4], qnum2[4];
  QUAD qconval;
#endif
 
  rslt = (CONST*)getitem(4, sizeof(CONST));
  BZERO(rslt, CONST, 1);
  rslt->id = AC_CONST;
  rslt->repeatc = 1;
  rslt->dtype = arg->dtype;  

  arg = eval_init_expr(arg);
  conval1 = arg->u1.conval;
  arg2 = arg->next;
 
 
  if (arg2->dtype == DT_INT8)
    error(S_0205_Illegal_specification_of_scale_factor, ERR_Warning, gbl.lineno, SYMNAME(arg2->u1.conval),
          "- Illegal specification of scale factor");
  
  i = (arg2->dtype == DT_INT8) ? CONVAL2G(arg2->u1.conval) : arg2->u1.conval;

  switch (size_of(arg->dtype)) {
  case 4:
     /* 8-bit exponent (127) to get an exponent value in the 
      * range -126 .. +127 */
    e = 127 + i;
    if (e < 0)
      e = 0;
    else if (e > 255)
      e = 255;
    
    /* calculate decimal value from it's IEEE 754 form*/
    conval2 = e << 23;
    xfmul(conval1, conval2, &conval);
    rslt->u1.conval = conval;
    break;

  case 8:
    e = 1023 + i;
    if (e < 0)
      e = 0;
    else if (e > 2047)
      e = 2047;

    inum1[0] = CONVAL1G(conval1);
    inum1[1] = CONVAL2G(conval1);

    inum2[0] = e << 20;
    inum2[1] = 0;
    xdmul(inum1, inum2, dconval);
    rslt->u1.conval = getcon(dconval, DT_DBLE);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case 16:
    e = 16383 + i;
    if (e < 0)
      e = 0;
    else if (e > 32767)
      e = 32767;

    qnum1[0] = CONVAL1G(conval1);
    qnum1[1] = CONVAL2G(conval1);
    qnum1[2] = CONVAL3G(conval1);
    qnum1[3] = CONVAL4G(conval1);

    qnum2[0] = e << 16;
    qnum2[1] = 0;
    qnum2[2] = 0;
    qnum2[3] = 0;
    xqmul(qnum1, qnum2, qconval);
    rslt->u1.conval = getcon(qconval, DT_QUAD);
    break;
#endif
  }

  return rslt;
}

static CONST *
eval_minval_or_maxval(CONST *arg, DTYPE dtype, int intrin)
{
  DTYPE elem_dt = array_element_dtype(dtype);
  DTYPE loc_dtype = DT_INT;
  CONST *array = eval_init_expr_item(arg);
  unsigned dim = 0; /* 0 means no DIM specified, otherwise in 1..ndims */
  CONST *mask = 0;
  bool back = FALSE;
  
  INDEX index;
  unsigned i;
  CONST *arg2;
  ADSC *adsc;
  int arr_ndims, extent, lwbd, upbd;

  while ((arg = arg->next)) {
    if (DT_ISLOG(arg->dtype)) { /* back */
      arg2 = eval_init_expr_item(arg);
      back = arg2->u1.conval;
    } else if (DT_ISINT(arg->dtype)) { /* dim */
      arg2 = eval_init_expr_item(arg);
      dim = arg2->u1.conval;
    } else {
      mask = eval_init_expr_item(arg);
      if (mask != 0 && mask->id == AC_ACONST)
        mask = mask->subc;
    }
  }

  if (mask == 0) {
    /* mask defaults to .true. */
    mask = (CONST*)getitem(4, sizeof(CONST));
    BZERO(mask, CONST, 1);
    mask->id = AC_CONST;
    mask->dtype = DT_LOG;
    mask->u1.conval = 1;
  }

  /* index contains the rank and extents of the array dtype */
  sb.root = sb.roottail = NULL;
  adsc = AD_DPTR(dtype);
  arr_ndims = index.ndims = AD_NUMDIM(adsc);
  for (i=1; i <= index.ndims; ++i) {
    lwbd = sb.dim[i].lowb = ad_val_of(AD_LWBD(adsc, i-1));
    upbd = sb.dim[i].upb = ad_val_of(AD_UPBD(adsc, i-1));
    sb.dim[i].mplyr = ad_val_of(AD_MLPYR(adsc, i));
    extent = upbd - lwbd + 1;
    index.extent[i] = extent;
    index.index[i] = 1;
  }
  
  sb.ndims = arr_ndims;

  return do_eval_minval_or_maxval(&index, elem_dt, loc_dtype, array,
                                  dim, mask, back, intrin);
}

static CONST *
eval_nint(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg;
  int conval;

  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT res[4];
    INT con1;
    DTYPE dtype1 = wrkarg->dtype;

    con1 = wrkarg->u1.conval;
    switch (DTY(dtype1)) {
    default:
      break;
    case TY_REAL:
      num1[0] = CONVAL2G(stb.flt0);
      if (xfcmp(con1, num1[0]) >= 0)
        xfadd(con1, CONVAL2G(stb.flthalf), &res[0]);
      else
        xfsub(con1, CONVAL2G(stb.flthalf), &res[0]);
      conval = cngcon(res[0], DT_REAL, DT_INT);
      break;
    case TY_DBLE:
      if (init_fold_const(OP_CMP, con1, stb.dbl0, DT_DBLE) >= 0)
        res[0] = init_fold_const(OP_ADD, con1, stb.dblhalf, DT_DBLE);
      else
        res[0] = init_fold_const(OP_SUB, con1, stb.dblhalf, DT_DBLE);
      conval = cngcon(res[0], DT_DBLE, DT_INT);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      if (init_fold_const(OP_CMP, con1, stb.quad0, DT_QUAD) >= 0)
        res[0] = init_fold_const(OP_ADD, con1, stb.quadhalf, DT_QUAD);
      else
        res[0] = init_fold_const(OP_SUB, con1, stb.quadhalf, DT_QUAD);
      conval = cngcon(res[0], DT_QUAD, DT_INT);
      break;
#endif
    }

    wrkarg->id = AC_CONST;
    wrkarg->dtype = DT_INT;
    wrkarg->repeatc = 1;
    wrkarg->u1.conval = conval;
  }
  return rslt;
}

INLINE static CONST *
eval_floor(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg;
  int conval;

  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT con1;
    int adjust;

    adjust = 0;
    con1 = wrkarg->u1.conval;
    switch (DTY(wrkarg->dtype)) {
    default:
      break;
    case TY_REAL:
      conval = cngcon(con1, DT_REAL, dtype);
      num1[0] = CONVAL2G(stb.flt0);
      if (xfcmp(con1, num1[0]) < 0) {
        con1 = cngcon(conval, dtype, DT_REAL);
        if (xfcmp(con1, wrkarg->u1.conval) != 0)
          adjust = 1;
      }
      break;
    case TY_DBLE:
      conval = cngcon(con1, DT_DBLE, dtype);
      if (init_fold_const(OP_CMP, con1, stb.dbl0, DT_DBLE) < 0) {
        con1 = cngcon(conval, dtype, DT_DBLE);
        if (init_fold_const(OP_CMP, con1, wrkarg->u1.conval, DT_DBLE) != 0)
          adjust = 1;
      }
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      conval = cngcon(con1, DT_QUAD, dtype);
      if (init_fold_const(OP_CMP, con1, stb.quad0, DT_QUAD) < 0) {
        con1 = cngcon(conval, dtype, DT_QUAD);
        if (init_fold_const(OP_CMP, con1, wrkarg->u1.conval, DT_QUAD) != 0)
          adjust = 1;
      }
      break;
#endif
    }
    if (adjust) {
      if (DT_ISWORD(dtype))
        conval--;
      else {
        num1[0] = 0;
        num1[1] = 1;
        con1 = getcon(num1, dtype);
        conval = init_fold_const(OP_SUB, conval, con1, dtype);
      }
    }
    wrkarg->u1.conval = conval;
    wrkarg->dtype = dtype;
    wrkarg->id = AC_CONST;
    wrkarg->repeatc = 1;
  }
  return rslt;
}

INLINE static CONST *
eval_ceiling(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg;
  int conval;

  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT con1;
    int adjust;

    adjust = 0;
    con1 = wrkarg->u1.conval;
    switch (DTY(wrkarg->dtype)) {
    default:
      break;
    case TY_REAL:
      conval = cngcon(con1, DT_REAL, dtype);
      num1[0] = CONVAL2G(stb.flt0);
      if (xfcmp(con1, num1[0]) > 0) {
        con1 = cngcon(conval, dtype, DT_REAL);
        if (xfcmp(con1, wrkarg->u1.conval) != 0)
          adjust = 1;
      }
      break;
    case TY_DBLE:
      conval = cngcon(con1, DT_DBLE, dtype);
      if (init_fold_const(OP_CMP, con1, stb.dbl0, DT_DBLE) > 0) {
        con1 = cngcon(conval, dtype, DT_DBLE);
        if (init_fold_const(OP_CMP, con1, wrkarg->u1.conval, DT_DBLE) != 0)
          adjust = 1;
      }
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      conval = cngcon(con1, DT_QUAD, dtype);
      if (init_fold_const(OP_CMP, con1, stb.quad0, DT_QUAD) > 0) {
        con1 = cngcon(conval, dtype, DT_QUAD);
        if (init_fold_const(OP_CMP, con1, wrkarg->u1.conval, DT_QUAD) != 0)
          adjust = 1;
      }
      break;
#endif
    }
    if (adjust) {
      if (DT_ISWORD(dtype))
        conval++;
      else {
        num1[0] = 0;
        num1[1] = 1;
        con1 = getcon(num1, dtype);
        conval = init_fold_const(OP_ADD, conval, con1, dtype);
      }
    }
    wrkarg->u1.conval = conval;
    wrkarg->dtype = dtype;
    wrkarg->id = AC_CONST;
    wrkarg->repeatc = 1;
  }
  return rslt;
}

static CONST *
eval_mod(CONST *arg, DTYPE dtype)
{
  CONST *rslt;
  CONST *arg1, *arg2;
  INT conval;
  arg1 = eval_init_expr_item(arg);
  arg2 = eval_init_expr_item(arg->next);
  rslt = clone_init_const_list(arg1, true);
  arg1 = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  arg2 = (arg2->id == AC_ACONST ? arg2->subc : arg2);
  for (; arg1; arg1 = arg1->next, arg2 = arg2->next) {
    /* mod(a,p) == a-int(a/p)*p  */
    INT num1[4], num2[4], num3[4];
    INT con1, con2, con3;
    con1 = arg1->u1.conval;
    con2 = arg2->u1.conval;
    /*
            conval1 = cngcon(arg1->u1.conval, arg1->dtype, dtype);
            conval2 = cngcon(arg2->u1.conval, arg2->dtype, dtype);
            conval3 = const_fold(OP_DIV, conval1, conval2, dtype);
            conval3 = cngcon(conval3, dtype, DT_INT8);
            conval3 = cngcon(conval3, DT_INT8, dtype);
            conval3 = const_fold(OP_MUL, conval3, conval2, dtype);
            conval3 = const_fold(OP_SUB, conval1, conval3, dtype);
            arg1->conval = conval3;
     */
    switch (DTY(arg1->dtype)) {
    case TY_REAL:
      xfdiv(con1, con2, &con3);
      con3 = cngcon(con3, DT_REAL, DT_INT8);
      con3 = cngcon(con3, DT_INT8, DT_REAL);
      xfmul(con3, con2, &con3);
      xfsub(con1, con3, &con3);
      conval = con3;
      break;
    case TY_DBLE:
      num1[0] = CONVAL1G(con1);
      num1[1] = CONVAL2G(con1);
      num2[0] = CONVAL1G(con2);
      num2[1] = CONVAL2G(con2);
      xddiv(num1, num2, num3);
      con3 = getcon(num3, DT_DBLE);
      con3 = cngcon(con3, DT_DBLE, DT_INT8);
      con3 = cngcon(con3, DT_INT8, DT_DBLE);
      num3[0] = CONVAL1G(con3);
      num3[1] = CONVAL2G(con3);
      xdmul(num3, num2, num3);
      xdsub(num1, num3, num3);
      conval = getcon(num3, DT_DBLE);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      num1[0] = CONVAL1G(con1);
      num1[1] = CONVAL2G(con1);
      num1[2] = CONVAL3G(con1);
      num1[3] = CONVAL4G(con1);
      num2[0] = CONVAL1G(con2);
      num2[1] = CONVAL2G(con2);
      num2[2] = CONVAL3G(con2);
      num2[3] = CONVAL4G(con2);
      xqdiv(num1, num2, num3);
      con3 = getcon(num3, DT_QUAD);
      con3 = cngcon(con3, DT_QUAD, DT_INT8);
      con3 = cngcon(con3, DT_INT8, DT_QUAD);
      num3[0] = CONVAL1G(con3);
      num3[1] = CONVAL2G(con3);
      num3[2] = CONVAL3G(con3);
      num3[3] = CONVAL4G(con3);
      xqmul(num3, num2, num3);
      xqsub(num1, num3, num3);
      conval = getcon(num3, DT_QUAD);
      break;
#endif
    case TY_CMPLX:
    case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
#endif
      error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,
            "Intrinsic not supported in initialization:", "mod");
      break;
    default:
      error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,
            "Intrinsic not supported in initialization:", "mod");
      break;
    }
    conval = cngcon(conval, arg1->dtype, dtype);
    arg1->u1.conval = conval;
    arg1->dtype = dtype;
    arg1->id = AC_CONST;
    arg1->repeatc = 1;
  }
  return rslt;
}

static CONST *
eval_repeat(CONST *arg, DTYPE dtype)
{
  CONST *rslt = NULL;
  CONST *arg1 = eval_init_expr_item(arg);
  CONST *arg2 = eval_init_expr_item(arg->next);
  int i, j, cvlen, newlen, result;
  int ncopies;
  char *p, *cp, *str;

  ncopies = arg2->u1.conval;
  newlen = size_of(dtype);
  cvlen = size_of(arg1->dtype);

  str = cp = getitem(0, newlen);
  j = ncopies;
  while (j-- > 0) {
    p = stb.n_base + CONVAL1G(arg1->u1.conval);
    i = cvlen;
    while (i-- > 0)
      *cp++ = *p++;
  }
  result = getstring(str, newlen);

  rslt = (CONST *)getitem(4, sizeof(CONST));
  BZERO(rslt, CONST, 1);
  rslt->id = AC_CONST;
  rslt->dtype = dtype;
  rslt->repeatc = 1;
  rslt->u1.conval = result;

  return rslt;
}

static CONST *
eval_len_trim(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg;
  char *p;
  int cvlen, result;

  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    p = stb.n_base + CONVAL1G(wrkarg->u1.conval);
    result = cvlen = size_of(wrkarg->dtype);
    p += cvlen - 1;
    /* skip trailing blanks */
    while (cvlen-- > 0) {
      if (*p-- != ' ')
        break;
      result--;
    }

    wrkarg->id = AC_CONST;
    wrkarg->dtype = DT_INT;
    wrkarg->repeatc = 1;
    wrkarg->u1.conval = result;
  }
  return rslt;
}

static CONST *
eval_selected_real_kind(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg;
  int r;
  int con;

  r = 4;

  wrkarg = eval_init_expr_item(arg);
  con = wrkarg->u1.conval; /* what about zero ?? */
  if (con <= 6)
    r = 4;
  else if (con <= 15)
    r = 8;
#ifdef TARGET_SUPPORTS_QUADFP
  else if (con <= MAX_EXP_OF_QMANTISSA)
    r = REAL_16;
#endif
  else
    r = -1;

  if (arg->next) {
    wrkarg = eval_init_expr_item(arg->next);
    con = wrkarg->u1.conval; /* what about zero ?? */
    if (con <= 37) {
      if (r > 0 && r < 4)
        r = 4;
    } else if (con <= 307) {
      if (r > 0 && r < 8)
        r = 8;
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (con <= MAX_EXP_QVALUE) {
      if (r > REAL_0 && r < REAL_16)
        r = REAL_16;
#endif
    } else {
      if (r > 0)
        r = 0;
      r -= 2;
    }
  }

  arg = arg->next;
  if (arg->next) {
    wrkarg = eval_init_expr_item(arg->next);;
    con = wrkarg->u1.conval;
    if (con != RADIX2) {
      if (con == NOT_GET_VAL && (wrkarg->sptr == 0)) {}
      else {
        r = NO_REAL;
      }
    }
  }

  rslt = (CONST *)getitem(4, sizeof(CONST));
  BZERO(rslt, CONST, 1);
  rslt->id = AC_CONST;
  rslt->dtype = DT_INT;
  rslt->repeatc = 1;
  rslt->u1.conval = r;

  return rslt;
}

static CONST *
eval_selected_int_kind(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg;
  int r;
  int con;

  wrkarg = eval_init_expr_item(arg);
  con = wrkarg->u1.conval;
  if (con > 18 || (con > 9 && XBIT(57, 2)))
    r = -1;
  else if (con > 9)
    r = 8;
  else if (con > 4)
    r = 4;
  else if (con > 2)
    r = 2;
  else
    r = 1;
  rslt->u1.conval = r;

  return rslt;
}

static CONST *
eval_selected_char_kind(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr(arg);
  int r;
  int con;

  con = rslt->u1.conval;
  if (sem_eq_str(con, "ASCII"))
    r = 1;
  else if (sem_eq_str(con, "DEFAULT"))
    return (CONST *)1;
  else
    r = -1;
  rslt = (CONST *)getitem(4, sizeof(CONST));
  BZERO(rslt, CONST, 1);
  rslt->id = AC_CONST;
  rslt->dtype = DT_INT;
  rslt->repeatc = 1;
  rslt->u1.conval = r;
  return rslt;
}

static CONST *
eval_scan(CONST *arg, DTYPE dtype)
{
  CONST *rslt = NULL;
  CONST *rslttail = NULL;
  CONST *c;
  CONST *wrkarg;
  int i, j;
  int l_string, l_set;
  char *p_string, *p_set;
  ISZ_T back = 0;

  assert(arg->next, "eval_scan: substring argument missing\n", 0, ERR_Fatal);
  wrkarg = eval_init_expr_item(arg->next);
  p_set = stb.n_base + CONVAL1G(wrkarg->u1.conval);
  l_set = size_of(wrkarg->dtype);

  if (arg->next->next) {
    wrkarg = eval_init_expr_item(arg->next->next);
    back = get_ival(wrkarg->dtype, wrkarg->u1.conval);
  }

  wrkarg = (arg->id == AC_ACONST ? arg->subc : arg);
  wrkarg = eval_init_expr_item(wrkarg);
  for (; wrkarg; wrkarg = wrkarg->next) {
    assert(wrkarg->id == AC_CONST, "eval_scan: non-constant argument\n", 0,
           ERR_Fatal);
    p_string = stb.n_base + CONVAL1G(wrkarg->u1.conval);
    l_string = size_of(wrkarg->dtype);

    c = (CONST *)getitem(4, sizeof(CONST));
    BZERO(c, CONST, 1);
    c->id = AC_CONST;
    c->dtype = DT_INT;
    c->repeatc = 1;

    if (back == 0) {
      for (i = 0; i < l_string; ++i)
        for (j = 0; j < l_set; ++j)
          if (p_set[j] == p_string[i]) {
            c->u1.conval = i + 1;
            goto addtolist;
          }
    } else {
      for (i = l_string - 1; i >= 0; --i)
        for (j = 0; j < l_set; ++j)
          if (p_set[j] == p_string[i]) {
            c->u1.conval = i + 1;
            goto addtolist;
          }
    }
    c->u1.conval = 0;

  addtolist:
    add_to_list(c, &rslt, &rslttail);
  }
  return rslt;
}

static CONST *
eval_verify(CONST *arg, DTYPE dtype)
{
  CONST *rslt = NULL;
  CONST *rslttail = NULL;
  CONST *c;
  CONST *wrkarg;
  int i, j;
  int l_string, l_set;
  char *p_string, *p_set;
  ISZ_T back = 0;

  assert(arg->next, "eval_verify: substring argument missing\n", 0, ERR_Fatal);
  wrkarg = eval_init_expr_item(arg->next);
  p_set = stb.n_base + CONVAL1G(wrkarg->u1.conval);
  l_set = size_of(wrkarg->dtype);

  if (arg->next->next) {
    wrkarg = eval_init_expr_item(arg->next->next);
    back = get_ival(wrkarg->dtype, wrkarg->u1.conval);
  }

  wrkarg = (arg->id == AC_ACONST ? arg->subc : arg);
  wrkarg = eval_init_expr_item(wrkarg);
  for (; wrkarg; wrkarg = wrkarg->next) {
    assert(wrkarg->id == AC_CONST, "eval_verify: non-constant argument\n", 0,
           ERR_Fatal);
    p_string = stb.n_base + CONVAL1G(wrkarg->u1.conval);
    l_string = size_of(wrkarg->dtype);

    c = (CONST *)getitem(4, sizeof(CONST));
    BZERO(c, CONST, 1);
    c->id = AC_CONST;
    c->dtype = DT_INT;
    c->repeatc = 1;
    c->u1.conval = 0;

    if (back == 0) {
      for (i = 0; i < l_string; ++i) {
        for (j = 0; j < l_set; ++j) {
          if (p_set[j] == p_string[i])
            goto contf;
        }
        c->u1.conval = i + 1;
        break;
      contf:;
      }
    } else {
      for (i = l_string - 1; i >= 0; --i) {
        for (j = 0; j < l_set; ++j) {
          if (p_set[j] == p_string[i])
            goto contb;
        }
        c->u1.conval = i + 1;
        break;
      contb:;
      }
    }

    add_to_list(c, &rslt, &rslttail);
  }
  return rslt;
}

static CONST *
eval_index(CONST *arg, DTYPE dtype)
{
  CONST *rslt = NULL;
  CONST *rslttail = NULL;
  CONST *c;
  CONST *wrkarg;
  int i, n;
  int l_string, l_substring;
  char *p_string, *p_substring;
  ISZ_T back = 0;

  assert(arg->next, "eval_index: substring argument missing\n", 0, ERR_Fatal);
  wrkarg = eval_init_expr_item(arg->next);
  p_substring = stb.n_base + CONVAL1G(wrkarg->u1.conval);
  l_substring = size_of(wrkarg->dtype);

  if (arg->next->next) {
    wrkarg = eval_init_expr_item(arg->next->next);
    back = get_ival(wrkarg->dtype, wrkarg->u1.conval);
  }

  wrkarg = (arg->id == AC_ACONST ? arg->subc : arg);
  wrkarg = eval_init_expr_item(wrkarg);
  for (; wrkarg; wrkarg = wrkarg->next) {
    assert(wrkarg->id == AC_CONST, "eval_index: non-constant argument\n", 0,
           ERR_Fatal);
    p_string = stb.n_base + CONVAL1G(wrkarg->u1.conval);
    l_string = size_of(wrkarg->dtype);

    c = (CONST *)getitem(4, sizeof(CONST));
    BZERO(c, CONST, 1);
    c->id = AC_CONST;
    c->dtype = DT_INT;
    c->repeatc = 1;

    n = l_string - l_substring;
    if (n < 0)
      c->u1.conval = 0;
    if (back == 0) {
      if (l_substring == 0)
        c->u1.conval = 1;
      for (i = 0; i <= n; ++i) {
        if (p_string[i] == p_substring[0] &&
            strncmp(p_string + i, p_substring, l_substring) == 0)
          c->u1.conval = i + 1;
      }
    } else {
      if (l_substring == 0)
        c->u1.conval = l_string + 1;
      for (i = n; i >= 0; --i) {
        if (p_string[i] == p_substring[0] &&
            strncmp(p_string + i, p_substring, l_substring) == 0)
          c->u1.conval = i + 1;
      }
    }
    add_to_list(c, &rslt, &rslttail);
  }
  return rslt;
}

static CONST *
eval_trim(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr(arg);
  const char *str;
  char *p, *cp;
  int i, cvlen, newlen;

  p = stb.n_base + CONVAL1G(rslt->u1.conval);
  cvlen = newlen = size_of(rslt->dtype);

  i = 0;
  p += cvlen - 1;
  /* skip trailing blanks */
  while (cvlen-- > 0) {
    if (*p-- != ' ')
      break;
    newlen--;
  }

  if (newlen == 0) {
    str = " ";
    rslt->u1.conval = getstring(str, strlen(str));
  } else {
    str = cp = getitem(0, newlen);
    i = newlen;
    cp += newlen - 1;
    p++;
    while (i-- > 0) {
      *cp-- = *p--;
    }
    rslt->u1.conval = getstring(str, newlen);
  }

  rslt->dtype = get_type(2, DTY(dtype), newlen);
  return rslt;
}

INLINE static CONST *
eval_adjustl(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr(arg);
  CONST *wrkarg;
  char *p, *cp, *str;
  char ch;
  int i, cvlen, origlen;

  wrkarg = rslt->id == AC_ACONST ? rslt->subc : rslt;
  for (; wrkarg; wrkarg = wrkarg->next) {
    assert(wrkarg->id == AC_CONST, "eval_adjustl: non-constant argument\n", 0,
           ERR_Fatal);
    p = stb.n_base + CONVAL1G(wrkarg->u1.conval);
    cvlen = size_of(wrkarg->dtype);
    origlen = cvlen;
    str = cp = getitem(0, cvlen + 1); /* +1 just in case cvlen is 0 */
    i = 0;
    /* left justify string - skip leading blanks */
    while (cvlen-- > 0) {
      ch = *p++;
      if (ch != ' ') {
        *cp++ = ch;
        break;
      }
      i++;
    }
    while (cvlen-- > 0)
      *cp++ = *p++;
    /* append blanks */
    while (i-- > 0)
      *cp++ = ' ';
    wrkarg->u1.conval = getstring(str, origlen);
  }

  return rslt;
}

static CONST *
eval_adjustr(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr(arg);
  CONST *wrkarg;
  char *p, *cp, *str;
  char ch;
  int i, cvlen, origlen;

  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    assert(wrkarg->id == AC_CONST, "eval_adjustl: non-constant argument\n", 0,
           ERR_Fatal);
    p = stb.n_base + CONVAL1G(wrkarg->u1.conval);
    origlen = cvlen = size_of(wrkarg->dtype);
    str = cp = getitem(0, cvlen + 1); /* +1 just in case cvlen is 0 */
    i = 0;
    p += cvlen - 1;
    cp += cvlen - 1;
    /* right justify string - skip trailing blanks */
    while (cvlen-- > 0) {
      ch = *p--;
      if (ch != ' ') {
        *cp-- = ch;
        break;
      }
      i++;
    }
    while (cvlen-- > 0)
      *cp-- = *p--;
    /* insert blanks */
    while (i-- > 0)
      *cp-- = ' ';
    wrkarg->u1.conval = getstring(str, origlen);
  }

  return rslt;
}

static CONST *
eval_shape(CONST *arg, DTYPE dtype)
{
  CONST *rslt;

  rslt = clone_init_const(arg, true);
  return rslt;
}

static CONST *
eval_size(CONST *arg, DTYPE dtype)
{
  CONST *arg1 = arg;
  CONST *arg2 = arg->next;
  CONST *arg3;
  CONST *rslt;
  int dim;
  int i;

  if ((arg3 = arg->next->next)) {
    arg3 = eval_init_expr_item(arg3);
    dim = arg3->u1.conval;
    arg2 = arg2->subc;
    for (i = 1; i < dim && arg2; i++) {
      arg2 = arg2->next;
    }
    rslt = clone_init_const(arg2, true);
  } else {
    rslt = clone_init_const(arg1, true);
  }

  return rslt;
}

static CONST *
eval_ul_bound(int ul_selector, CONST *arg, DTYPE dtype)
{
  CONST *arg1 = arg;
  CONST *arg2;
  int arg2const;
  CONST *rslt;
  ADSC *adsc = AD_DPTR(arg1->dtype);
  int rank = AD_UPBD(adsc, 0);
  int i;

  if (arg->next) {
    arg2 = eval_init_expr_item(arg->next);
    arg2const = arg2->u1.conval;
    if (arg2const > rank) {
      error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,
            "DIM argument greater than the array rank", CNULL);
      return CONST_ERR(dtype);
    }
    rslt = arg1->subc;
    for (i = 1; rslt && i < arg2const; i++) {
      rslt = rslt->next;
    }
    rslt = clone_init_const(rslt, true);
  } else {
    rslt = clone_init_const(arg1, true);
  }
  return rslt;
}

static int
copy_initconst_to_array(CONST **arr, CONST *c, int count)
{
  int i;
  int acnt;
  CONST *acl;

  for (i = 0; i < count;) {
    if (c == NULL)
      break;
    switch (c->id) {
    case AC_ACONST:
      acnt = copy_initconst_to_array(arr, c->subc, count - i);
      /* MORE: count - i??? */
      i += acnt;
      arr += acnt;
      break;
    case AC_CONST:
      acl = *arr = clone_init_const(c, true);
      if (acl->repeatc > 1) {
        arr += acl->repeatc;
        i += acl->repeatc;
      } else {
        arr++;
        i++;
      }
      break;
    default:
      interr("copy_initconst_to_array: unexpected const type", c->id,
             ERR_Severe);
      return count;
    }
    c = c->next;
  }
  return i;
}

static CONST *
eval_reshape(CONST *arg, DTYPE dtype, LOGICAL transpose)
{
  CONST *srclist = eval_init_expr_item(arg);
  CONST *tacl;
  CONST *pad = NULL;
  CONST *wrklist = NULL;
  CONST *orderarg = NULL;
  CONST **old_val = NULL;
  CONST **new_val = NULL;
  CONST *c = NULL;
  ADSC *adsc = AD_DPTR(dtype);
  int *new_index;
  int src_sz, dest_sz;
  int rank;
  int order[7];
  int lwb[7];
  int upb[7];
  int mult[7];
  int i;
  int count;
  int sz;

  if (arg->next) {
    eval_init_expr_item(arg->next);

    if (arg->next->next) {
      pad = arg->next->next;
      if (pad->id != AC_CONST) {
        pad = eval_init_expr_item(pad);
      }
      if (arg->next->next->next && arg->next->next->next->id != AC_CONST) {
        orderarg = eval_init_expr_item(arg->next->next->next);
      }
    }
  }
  src_sz = ad_val_of(AD_NUMELM(AD_DPTR(arg->dtype)));
  dest_sz = ad_val_of(AD_NUMELM(adsc));

  rank = AD_NUMDIM(adsc);
  sz = 1;
  for (i = 0; i < rank; i++) {
    upb[i] = ad_val_of(AD_UPBD(adsc, i));
    lwb[i] = 0;
    mult[i] = sz;
    sz *= upb[i];
  }

  if (orderarg == NULL) {
    if (transpose) {
      order[0] = 1;
      order[1] = 0;
    } else {
      if (src_sz == dest_sz) {
        return srclist;
      }
      for (i = 0; i < rank; i++) {
        order[i] = i;
      }
    }
  } else {
    bool out_of_order;

    out_of_order = false;
    c = (orderarg->id == AC_ACONST ? orderarg->subc : orderarg);
    for (i = 0; c && i < rank; c = c->next, i++) {
      order[i] =
          DT_ISWORD(c->dtype) ? c->u1.conval - 1 : ad_val_of(c->u1.conval) - 1;
      if (order[i] != i)
        out_of_order = true;
    }
    if (!out_of_order && src_sz == dest_sz) {
      return srclist;
    }
  }

  NEW(old_val, CONST *, dest_sz);
  if (old_val == NULL)
    return CONST_ERR(dtype);
  BZERO(old_val, CONST *, dest_sz);
  NEW(new_val, CONST *, dest_sz);
  if (new_val == NULL) {
    return CONST_ERR(dtype);
  }
  BZERO(new_val, CONST *, dest_sz);
  NEW(new_index, int, dest_sz);
  if (new_index == NULL) {
    return CONST_ERR(dtype);
  }
  BZERO(new_index, int, dest_sz);

  count = dest_sz > src_sz ? src_sz : dest_sz;
  wrklist = srclist->id == AC_ACONST ? srclist->subc : srclist;
  (void)copy_initconst_to_array(old_val, wrklist, count);

  if (dest_sz > src_sz) {
    count = dest_sz - src_sz;
    wrklist = pad->id == AC_ACONST ? pad->subc : pad;
    while (count > 0) {
      i = copy_initconst_to_array(old_val + src_sz, wrklist, count);
      count -= i;
      src_sz += i;
    }
  }

  /* index to access source in linear order */
  i = 0;
  while (true) {
    int index; /* index where to store each element of new val */
    int j;

    index = 0;
    for (j = 0; j < rank; j++)
      index += lwb[j] * mult[j];

    new_index[index] = i;

    /* update loop indices */
    for (j = 0; j < rank; j++) {
      int loop;
      loop = order[j];
      lwb[loop]++;
      if (lwb[loop] < upb[loop])
        break;
      lwb[loop] = 0; /* reset and go on to the next loop */
    }
    if (j >= rank)
      break;
    i++;
  }

  for (i = 0; i < dest_sz; i++) {
    CONST *tail;
    int idx, start, end;
    int index = new_index[i];
    if (old_val[index]) {
      if (old_val[index]->repeatc <= 1) {
        new_val[i] = old_val[index];
        new_val[i]->id = AC_CONVAL;
      } else {
        idx = index + 1;
        start = i;
        end = old_val[index]->repeatc - 1;
        while (new_index[++start] == idx) {
          ++idx;
          --end;
          if (end <= 0 || start > dest_sz - 1)
            break;
        }
        old_val[index]->next = NULL;
        tacl = clone_init_const(old_val[index], true);
        tacl->repeatc = idx - index;
        tacl->id = AC_CONVAL;
        old_val[index]->repeatc = index - (idx - index);
        new_val[i] = tacl;
      }
    } else {
      tail = old_val[index];
      idx = index;
      while (tail == NULL && idx >= 0) {
        tail = old_val[idx--];
      }
      tail->next = NULL;
      tacl = clone_init_const(tail, true);
      start = i;
      end = tail->repeatc - 1;
      idx = index + 1;
      while (new_index[++start] == idx) {
        ++idx;
        --end;
        if (end <= 0 || start > dest_sz - 1)
          break;
      }
      tail->repeatc = index - (idx - index);
      tacl->repeatc = idx - index;
      tacl->id = AC_CONVAL;
      new_val[i] = tacl;
    }
  }
  tacl = new_val[0];
  for (i = 0; i < dest_sz - 1; ++i) {
    if (new_val[i + 1] == NULL) {
      continue;
    } else {
      tacl->next = new_val[i + 1];
      tacl = new_val[i + 1];
    }
  }
  if (new_val[dest_sz - 1])
    (new_val[dest_sz - 1])->next = NULL;
  srclist = *new_val;

  FREE(old_val);
  FREE(new_val);
  FREE(new_index);

  return srclist;
}

/* Store the value 'conval' of type 'dtype' into 'destination'. */
static void
transfer_store(INT conval, DTYPE dtype, char *destination)
{
  int *dest = (int *)destination;
  INT real, imag;

  if (DT_ISWORD(dtype)) {
    dest[0] = conval;
    return;
  }

  switch (DTY(dtype)) {
  case TY_DWORD:
  case TY_INT8:
  case TY_LOG8:
  case TY_DBLE:
    dest[0] = CONVAL2G(conval);
    dest[1] = CONVAL1G(conval);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    dest[0] = CONVAL4G(conval);
    dest[1] = CONVAL3G(conval);
    dest[2] = CONVAL2G(conval);
    dest[3] = CONVAL1G(conval);
    break;
#endif

  case TY_CMPLX:
    dest[0] = CONVAL1G(conval);
    dest[1] = CONVAL2G(conval);
    break;

  case TY_DCMPLX:
    real = CONVAL1G(conval);
    imag = CONVAL2G(conval);
    dest[0] = CONVAL2G(real);
    dest[1] = CONVAL1G(real);
    dest[2] = CONVAL2G(imag);
    dest[3] = CONVAL1G(imag);
    break;

  case TY_CHAR:
    memcpy(dest, stb.n_base + CONVAL1G(conval), size_of(dtype));
    break;

  default:
    interr("transfer_store: unexpected dtype", dtype, ERR_Severe);
  }
}

/* Get a value of type 'dtype' from buffer 'source'. */
static INT
transfer_load(DTYPE dtype, char *source)
{
  int *src = (int *)source;
  INT num[4], real[2], imag[2];

  if (DT_ISWORD(dtype))
    return src[0];

  switch (DTY(dtype)) {
  case TY_DWORD:
  case TY_INT8:
  case TY_LOG8:
  case TY_DBLE:
    num[1] = src[0];
    num[0] = src[1];
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    num[3] = src[0];
    num[2] = src[1];
    num[1] = src[2];
    num[0] = src[3];
    break;
#endif

  case TY_CMPLX:
    num[0] = src[0];
    num[1] = src[1];
    break;

  case TY_DCMPLX:
    real[1] = src[0];
    real[0] = src[1];
    imag[1] = src[2];
    imag[0] = src[3];
    num[0] = getcon(real, DT_DBLE);
    num[1] = getcon(imag, DT_DBLE);
    break;

  case TY_CHAR:
    return getstring(source, size_of(dtype));

  default:
    interr("transfer_load: unexpected dtype", dtype, ERR_Severe);
  }

  return getcon(num, dtype);
}

INLINE static CONST *
eval_transfer(CONST *arg, DTYPE dtype)
{
  CONST *src = eval_init_expr(arg);
  CONST *rslt;
  int avail;
  char value[256];
  char *buffer = value;
  char *bp;
  INT pad;

  /* Find type and size of the source and result. */
  DTYPE sdtype = DDTG(src->dtype);
  int ssize = size_of(sdtype);
  DTYPE rdtype = DDTG(dtype);
  int rsize = size_of(rdtype);

  /* Be sure we have enough space. */
  int need = (rsize > ssize ? rsize : ssize) * 2;

  if (sizeof(value) < (size_t)need) {
    NEW(buffer, char, need);
    if (buffer == NULL)
      return CONST_ERR(dtype);
  }

  /* Get pad value in case we have to fill. */
  if (DTY(sdtype) == TY_CHAR)
    memset(buffer, ' ', ssize);
  else
    BZERO(buffer, char, ssize);
  pad = transfer_load(sdtype, buffer);

  if (src->id == AC_ACONST)
    src = src->subc;
  bp = buffer;
  avail = 0;
  if (DTY(dtype) != TY_ARRAY) {
    /* Result is scalar. */
    while (avail < rsize) {
      if (src) {
        transfer_store(src->u1.conval, sdtype, bp);
        src = src->next;
      } else
        transfer_store(pad, sdtype, bp);
      bp += ssize;
      avail += ssize;
    }
    rslt = (CONST *)getitem(4, sizeof(CONST));
    BZERO(rslt, CONST, 1);
    rslt->id = AC_CONST;
    rslt->dtype = rdtype;
    rslt->u1.conval = transfer_load(rdtype, buffer);
    rslt->repeatc = 1;
  } else {
    /* Result is array. */
    CONST *root, **current;
    ISZ_T i, nelem;
    int j, cons;

    cons = AD_NUMELM(AD_DPTR(dtype));
    assert(STYPEG(cons) == ST_CONST, "eval_transfer: nelem not const", dtype,
           ERR_Severe);
    nelem = ad_val_of(cons);
    root = NULL;
    current = &root;
    for (i = 0; i < nelem; i++) {
      while (avail < rsize) {
        if (src) {
          transfer_store(src->u1.conval, sdtype, bp);
          src = src->next;
        } else {
          transfer_store(pad, sdtype, bp);
        }
        bp += ssize;
        avail += ssize;
      }
      rslt = (CONST *)getitem(4, sizeof(CONST));
      BZERO(rslt, CONST, 1);
      rslt->id = AC_CONST;
      rslt->dtype = rdtype;
      rslt->u1.conval = transfer_load(rdtype, buffer);
      rslt->repeatc = 1;
      *current = rslt;
      current = &(rslt->next);
      bp -= rsize;
      avail -= rsize;
      for (j = 0; j < avail; j++)
        buffer[j] = buffer[rsize + j];
    }
    rslt = (CONST *)getitem(4, sizeof(CONST));
    BZERO(rslt, CONST, 1);
    rslt->id = AC_ACONST;
    rslt->dtype = dtype;
    rslt->subc = root;
    rslt->repeatc = 1;
  }

  if (buffer != value)
    FREE(buffer);
  return rslt;
}

INLINE static CONST *
eval_sqrt(CONST *arg, DTYPE dtype)
{
  CONST *rslt = eval_init_expr_item(arg);
  CONST *wrkarg;
  INT conval;

  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT res[4];
    INT con1;

    con1 = wrkarg->u1.conval;
    switch (DTY(wrkarg->dtype)) {
    case TY_REAL:
      xfsqrt(con1, &res[0]);
      conval = res[0];
      break;
    case TY_DBLE:
      num1[0] = CONVAL1G(con1);
      num1[1] = CONVAL2G(con1);
      xdsqrt(num1, res);
      conval = getcon(res, DT_DBLE);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      num1[0] = CONVAL1G(con1);
      num1[1] = CONVAL2G(con1);
      num1[2] = CONVAL3G(con1);
      num1[3] = CONVAL4G(con1);
      xqsqrt(num1, res);
      conval = getcon(res, DT_QUAD);
      break;
#endif
    case TY_CMPLX:
    case TY_DCMPLX:
      /*
          a = sqrt(real**2 + imag**2);  "hypot(real,imag)
          if (a == 0) {
              x = 0;
              y = 0;
          }
          else if (real > 0) {
              x = sqrt(0.5 * (a + real));
              y = 0.5 * (imag / x);
          }
          else {
              y = sqrt(0.5 * (a - real));
              if (imag < 0)
                  y = -y;
              x = 0.5 * (imag / y);
          }
          res.real = x;
          res.imag = y;
      */

      error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno, "Intrinsic not supported in initialization:",
            "sqrt");
      break;
    default:
      error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno, "Intrinsic not supported in initialization:",
            "sqrt");
      break;
    }
    conval = cngcon(conval, wrkarg->dtype, dtype);
    wrkarg->u1.conval = conval;
    wrkarg->dtype = dtype;
    wrkarg->id = AC_CONST;
    wrkarg->repeatc = 1;
  }
  return rslt;
}

/*----------------------------------------------------------------------------*/

#ifdef TARGET_SUPPORTS_QUADFP
#define FPINTRIN1(iname, ent, fscutil, dscutil, qscutil)                       \
  static CONST *ent(CONST *arg, DTYPE dtype)                                   \
  {                                                                            \
    CONST *rslt = eval_init_expr_item(arg);                                    \
    CONST *wrkarg;                                                             \
    INT conval;                                                                \
    wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);                      \
    for (; wrkarg; wrkarg = wrkarg->next) {                                    \
      INT num1[4];                                                             \
      INT res[4];                                                              \
      INT con1;                                                                \
      con1 = wrkarg->u1.conval;                                                \
      switch (DTY(wrkarg->dtype)) {                                            \
      case TY_REAL:                                                            \
        fscutil(con1, &res[0]);                                                \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        dscutil(num1, res);                                                    \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_QUAD:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num1[2] = CONVAL3G(con1);                                              \
        num1[3] = CONVAL4G(con1);                                              \
        qscutil(num1, res);                                                    \
        conval = getcon(res, DT_QUAD);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
      case TY_QCMPLX:                                                          \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      default:                                                                 \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, wrkarg->dtype, dtype);                           \
      wrkarg->u1.conval = conval;                                              \
      wrkarg->dtype = dtype;                                                   \
      wrkarg->id = AC_CONST;                                                   \
      wrkarg->repeatc = 1;                                                     \
    }                                                                          \
    return rslt;                                                               \
  }
#else
#define FPINTRIN1(iname, ent, fscutil, dscutil, qscutil)                       \
  static CONST *ent(CONST *arg, DTYPE dtype)                                   \
  {                                                                            \
    CONST *rslt = eval_init_expr_item(arg);                                    \
    CONST *wrkarg;                                                             \
    INT conval;                                                                \
    wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);                      \
    for (; wrkarg; wrkarg = wrkarg->next) {                                    \
      INT num1[4];                                                             \
      INT res[4];                                                              \
      INT con1;                                                                \
      con1 = wrkarg->u1.conval;                                                \
      switch (DTY(wrkarg->dtype)) {                                            \
      case TY_REAL:                                                            \
        fscutil(con1, &res[0]);                                                \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        dscutil(num1, res);                                                    \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      default:                                                                 \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, wrkarg->dtype, dtype);                           \
      wrkarg->u1.conval = conval;                                              \
      wrkarg->dtype = dtype;                                                   \
      wrkarg->id = AC_CONST;                                                   \
      wrkarg->repeatc = 1;                                                     \
    }                                                                          \
    return rslt;                                                               \
  }
#endif

FPINTRIN1("exp", eval_exp, xfexp, xdexp, xqexp)

FPINTRIN1("log", eval_log, xflog, xdlog, xqlog)

FPINTRIN1("log10", eval_log10, xflog10, xdlog10, xqlog10)

FPINTRIN1("sin", eval_sin, xfsin, xdsin, xqsin)

FPINTRIN1("cos", eval_cos, xfcos, xdcos, xqcos)

FPINTRIN1("tan", eval_tan, xftan, xdtan, xqtan)

FPINTRIN1("asin", eval_asin, xfasin, xdasin, xqasin)

FPINTRIN1("acos", eval_acos, xfacos, xdacos, xqacos)

FPINTRIN1("atan", eval_atan, xfatan, xdatan, xqatan)

#ifdef TARGET_SUPPORTS_QUADFP
#define FPINTRIN2(iname, ent, fscutil, dscutil, qscutil)                       \
  static CONST *ent(CONST *arg, DTYPE dtype)                                   \
  {                                                                            \
    CONST *rslt;                                                               \
    CONST *arg1, *arg2;                                                        \
    INT conval;                                                                \
    arg1 = eval_init_expr_item(arg);                                           \
    arg2 = eval_init_expr_item(arg->next);                                     \
    rslt = clone_init_const_list(arg1, true);                                  \
    arg1 = (rslt->id == AC_ACONST ? rslt->subc : rslt);                        \
    arg2 = (arg2->id == AC_ACONST ? arg2->subc : arg2);                        \
    for (; arg1; arg1 = arg1->next, arg2 = arg2->next) {                       \
      INT num1[4], num2[4];                                                    \
      INT res[4];                                                              \
      INT con1, con2;                                                          \
      con1 = arg1->u1.conval;                                                  \
      con2 = arg2->u1.conval;                                                  \
      switch (DTY(arg1->dtype)) {                                              \
      case TY_REAL:                                                            \
        fscutil(con1, con2, &res[0]);                                          \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num2[0] = CONVAL1G(con2);                                              \
        num2[1] = CONVAL2G(con2);                                              \
        dscutil(num1, num2, res);                                              \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_QUAD:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num1[2] = CONVAL3G(con1);                                              \
        num1[3] = CONVAL4G(con1);                                              \
        num2[0] = CONVAL1G(con2);                                              \
        num2[1] = CONVAL2G(con2);                                              \
        num2[2] = CONVAL3G(con2);                                              \
        num2[3] = CONVAL4G(con2);                                              \
        qscutil(num1, num2, res);                                              \
        conval = getcon(res, DT_QUAD);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
      case TY_QCMPLX:                                                          \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      default:                                                                 \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, arg1->dtype, dtype);                             \
      arg1->u1.conval = conval;                                                \
      arg1->dtype = dtype;                                                     \
      arg1->id = AC_CONST;                                                     \
      arg1->repeatc = 1;                                                       \
    }                                                                          \
    return rslt;                                                               \
  }
#else
#define FPINTRIN2(iname, ent, fscutil, dscutil, qscutil)                       \
  static CONST *ent(CONST *arg, DTYPE dtype)                                   \
  {                                                                            \
    CONST *rslt;                                                               \
    CONST *arg1, *arg2;                                                        \
    INT conval;                                                                \
    arg1 = eval_init_expr_item(arg);                                           \
    arg2 = eval_init_expr_item(arg->next);                                     \
    rslt = clone_init_const_list(arg1, true);                                  \
    arg1 = (rslt->id == AC_ACONST ? rslt->subc : rslt);                        \
    arg2 = (arg2->id == AC_ACONST ? arg2->subc : arg2);                        \
    for (; arg1; arg1 = arg1->next, arg2 = arg2->next) {                       \
      INT num1[4], num2[4];                                                    \
      INT res[4];                                                              \
      INT con1, con2;                                                          \
      con1 = arg1->u1.conval;                                                  \
      con2 = arg2->u1.conval;                                                  \
      switch (DTY(arg1->dtype)) {                                              \
      case TY_REAL:                                                            \
        fscutil(con1, con2, &res[0]);                                          \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num2[0] = CONVAL1G(con2);                                              \
        num2[1] = CONVAL2G(con2);                                              \
        dscutil(num1, num2, res);                                              \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      default:                                                                 \
        error(S_0155_OP1_OP2, ERR_Severe, gbl.lineno,                          \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, arg1->dtype, dtype);                             \
      arg1->u1.conval = conval;                                                \
      arg1->dtype = dtype;                                                     \
      arg1->id = AC_CONST;                                                     \
      arg1->repeatc = 1;                                                       \
    }                                                                          \
    return rslt;                                                               \
  }
#endif

FPINTRIN2("atan2", eval_atan2, xfatan2, xdatan2, xqatan2)

INLINE static CONST *
eval_merge(CONST *arg, DTYPE dtype)
{
  CONST *tsource = eval_init_expr_item(arg);
  CONST *fsource = eval_init_expr_item(arg->next);
  CONST *mask = eval_init_expr_item(arg->next->next);
  CONST *result = clone_init_const_list(tsource, true);
  CONST *r = result;
  if (tsource->id == AC_ACONST)
    tsource = tsource->subc;
  if (fsource->id == AC_ACONST)
    fsource = fsource->subc;
  if (mask->id == AC_ACONST)
    mask = mask->subc;
  if (r->id == AC_ACONST)
    r = r->subc;
  for (; r != 0; r = r->next) {
    int mask_val = mask->u1.conval;
    int cond = DT_ISWORD(mask->dtype) ? mask_val : CONVAL2G(mask_val);
    r->u1.conval = cond ? tsource->u1.conval : fsource->u1.conval;
    r->dtype = dtype;
    tsource = tsource->next;
    fsource = fsource->next;
    mask = mask->next;
  }
  return result;
}

/*---------------------------------------------------------------------*/

static void
mk_cmp(CONST *c, int op, INT l_conval, INT r_conval, DTYPE rdtype, DTYPE dt)
{
  switch (get_ast_op(op)) {
  case OP_EQ:
  case OP_GE:
  case OP_GT:
  case OP_LE:
  case OP_LT:
  case OP_NE:
    l_conval =
        init_fold_const(OP_CMP, l_conval, r_conval, rdtype);
    switch (get_ast_op(op)) {
    case OP_EQ:
      l_conval = l_conval == 0;
      break;
    case OP_GE:
      l_conval = l_conval >= 0;
      break;
    case OP_GT:
      l_conval = l_conval > 0;
      break;
    case OP_LE:
      l_conval = l_conval <= 0;
      break;
    case OP_LT:
      l_conval = l_conval < 0;
      break;
    case OP_NE:
      l_conval = l_conval != 0;
      break;
    }
    l_conval = l_conval ? SCFTN_TRUE : SCFTN_FALSE;
    c->u1.conval = l_conval;
    break;
  case OP_LEQV:
    l_conval =
        init_fold_const(OP_CMP, l_conval, r_conval, rdtype);
     c->u1.conval = l_conval == 0;
    break;
  case OP_LNEQV:
    l_conval =
        init_fold_const(OP_CMP, l_conval, r_conval, rdtype);
    c->u1.conval = l_conval != 0;
    break;
  case OP_LOR:
    c->u1.conval = l_conval | r_conval;
    break;
  case OP_LAND:
    c->u1.conval = l_conval & r_conval;
    break;
  case OP_XTOI:
  case OP_XTOK:
  default:
    c->u1.conval = init_fold_const(get_ast_op(op), l_conval, r_conval, dt);
    break;
  }
}

static CONST *
eval_init_op(int op, CONST *lop, DTYPE ldtype, CONST *rop, DTYPE rdtype,
             SPTR sptr, DTYPE dtype)
{
  CONST *root = NULL;
  CONST *roottail = NULL;
  CONST *c;
  CONST *cur_lop;
  CONST *cur_rop;
  DTYPE dt = DDTG(dtype);
  DTYPE e_dtype;
  int i;
  ISZ_T l_repeatc;
  ISZ_T r_repeatc;
  INT l_conval;
  INT r_conval;
  int lsptr;
  int rsptr;
  char *s;
  int llen;
  int rlen;

  if (op == AC_NEG || op == AC_LNOT) {
    cur_lop = lop->id == AC_ACONST ? lop->subc : lop;
    for (; cur_lop; cur_lop = cur_lop->next) {
      c = (CONST *)getitem(4, sizeof(CONST));
      BZERO(c, CONST, 1);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = 1;
      l_conval = cur_lop->u1.conval;
      if (dt != cur_lop->dtype) {
        l_conval = cngcon(l_conval, DDTG(cur_lop->dtype), dt);
      }
      if (op == AC_LNOT)
        c->u1.conval = ~(l_conval);
      else
        c->u1.conval = init_negate_const(l_conval, dt);
      add_to_list(c, &root, &roottail);
    }
  } else if (op == AC_ARRAYREF) {
    root = eval_const_array_section(lop, ldtype, dtype);
  } else if (op == AC_CONV) {
    cur_lop = lop->id == AC_ACONST ? lop->subc : lop;
    l_repeatc = cur_lop->repeatc;
    for (; cur_lop;) {
      c = (CONST *)getitem(4, sizeof(CONST));
      BZERO(c, CONST, 1);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = 1;
      c->u1.conval = cngcon(cur_lop->u1.conval, DDTG(ldtype), DDTG(dtype));
      add_to_list(c, &root, &roottail);
      if (--l_repeatc <= 0) {
        cur_lop = cur_lop->next;
        if (cur_lop) {
          l_repeatc = cur_lop->repeatc;
        }
      }
    }
  } else if (op == AC_MEMBR_SEL) {
    c = eval_init_expr(lop);
    for (i = rop->u1.conval, cur_lop = c->subc; i > 0 && cur_lop;
         i--, cur_lop = cur_lop->next)
      ;
    if (!cur_lop) {
      interr("Malformed member select operator", op, ERR_Severe);
      return CONST_ERR(dtype);
    }
    root = clone_init_const(cur_lop, true);
    root->next = NULL;
  } else if (op == AC_CAT && DTY(ldtype) != TY_ARRAY &&
             DTY(rdtype) != TY_ARRAY) {
    lsptr = lop->u1.conval;
    rsptr = rop->u1.conval;
    llen = size_of(DDTG(ldtype));
    rlen = size_of(DDTG(rdtype));
    s = getitem(0, llen + rlen);
    BCOPY(s, stb.n_base + CONVAL1G(lsptr), char, llen);
    BCOPY(s + llen, stb.n_base + CONVAL1G(rsptr), char, rlen);

    c = (CONST *)getitem(4, sizeof(CONST));
    BZERO(c, CONST, 1);
    c->id = AC_CONST;
    c->dtype = get_type(2, TY_CHAR, llen + rlen); /* should check char type */
    c->repeatc = 1;
    c->u1.conval = c->sptr = getstring(s, llen + rlen);
    add_to_list(c, &root, &roottail);
  } else if (op == AC_INTR_CALL) {
    int intrin = lop->u1.conval;
    switch (lop->u1.conval) {
    case AC_I_adjustl:
      root = eval_adjustl(rop, dtype);
      break;
    case AC_I_adjustr:
      root = eval_adjustr(rop, dtype);
      break;
    case AC_I_char:
      root = eval_char(rop, dtype);
      break;
    case AC_I_ichar:
      root = eval_ichar(rop, dtype);
      break;
    case AC_I_index:
      root = eval_index(rop, dtype);
      break;
    case AC_I_int:
      root = eval_int(rop, dtype);
      break;
    case AC_I_ishft:
      root = eval_ishft(rop, dtype);
      break;
    case AC_I_len_trim:
      root = eval_len_trim(rop, dtype);
      break;
    case AC_I_ubound:
    case AC_I_lbound:
      root = eval_ul_bound(lop->u1.conval, rop, dtype);
      break;
    case AC_I_min:
      root = eval_min(rop, dtype);
      break;
    case AC_I_max:
      root = eval_max(rop, dtype);
      break;
    case AC_I_nint:
      root = eval_nint(rop, dtype);
      break;
    case AC_I_fltconvert:
      root = eval_fltconvert(rop, dtype);
      break;
    case AC_I_repeat:
      root = eval_repeat(rop, dtype);
      break;
    case AC_I_transpose:
      root = eval_reshape(rop, dtype, /*transpose*/ TRUE);
      break;
    case AC_I_reshape:
      root = eval_reshape(rop, dtype, /*transpose*/ FALSE);
      break;
    case AC_I_selected_int_kind:
      root = eval_selected_int_kind(rop, dtype);
      break;
    case AC_I_selected_real_kind:
      root = eval_selected_real_kind(rop, dtype);
      break;
    case AC_I_selected_char_kind:
      root = eval_selected_char_kind(rop, dtype);
      break;
    case AC_I_scan:
      root = eval_scan(rop, dtype);
      break;
    case AC_I_shape:
      root = eval_shape(rop, dtype);
      break;
    case AC_I_size:
      root = eval_size(rop, dtype);
      break;
    case AC_I_trim:
      root = eval_trim(rop, dtype);
      break;
    case AC_I_verify:
      root = eval_verify(rop, dtype);
      break;
    case AC_I_floor:
      root = eval_floor(rop, dtype);
      break;
    case AC_I_ceiling:
      root = eval_ceiling(rop, dtype);
      break;
    case AC_I_mod:
      root = eval_mod(rop, dtype);
      break;
    case AC_I_null:
      root = eval_null(rop, dtype);
      break;
    case AC_I_transfer:
      root = eval_transfer(rop, dtype);
      break;
    case AC_I_sqrt:
      root = eval_sqrt(rop, dtype);
      break;
    case AC_I_exp:
      root = eval_exp(rop, dtype);
      break;
    case AC_I_log:
      root = eval_log(rop, dtype);
      break;
    case AC_I_log10:
      root = eval_log10(rop, dtype);
      break;
    case AC_I_sin:
      root = eval_sin(rop, dtype);
      break;
    case AC_I_cos:
      root = eval_cos(rop, dtype);
      break;
    case AC_I_tan:
      root = eval_tan(rop, dtype);
      break;
    case AC_I_asin:
      root = eval_asin(rop, dtype);
      break;
    case AC_I_acos:
      root = eval_acos(rop, dtype);
      break;
    case AC_I_atan:
      root = eval_atan(rop, dtype);
      break;
    case AC_I_atan2:
      root = eval_atan2(rop, dtype);
      break;
    case AC_I_abs:
      root = eval_abs(rop, dtype);
      break;
    case AC_I_iand:
      root = eval_iand(rop, dtype);
      break;
    case AC_I_ior:
      root = eval_ior(rop, dtype);
      break;
    case AC_I_ieor:
      root = eval_ieor(rop, dtype);
      break;
    case AC_I_merge:
      root = eval_merge(rop, dtype);
      break;
    case AC_I_maxval:
    case AC_I_minval:
    case AC_I_maxloc:
    case AC_I_minloc:
      root = eval_minval_or_maxval(rop, rdtype, intrin);
      break;
    case AC_I_scale:
      root = eval_scale(rop, dtype);
      break;
    default:
      interr("eval_init_op(dinit.c): intrinsic not supported in "
             "initialization", lop->u1.conval, ERR_Severe);
      return CONST_ERR(dtype);
    }
  } else if (DTY(ldtype) == TY_ARRAY && DTY(rdtype) == TY_ARRAY) {
    /* array <binop> array */
    cur_lop = lop->id == AC_ACONST ? lop->subc : lop;
    cur_rop = rop->id == AC_ACONST ? rop->subc : rop;
    l_repeatc = cur_lop->repeatc;
    r_repeatc = cur_rop->repeatc;
    e_dtype = DDTG(dtype) != DT_LOG ? DDTG(dtype) : DDTG(lop->dtype);
    if (op == AC_CAT) {
      for (; cur_rop && cur_lop;) {
        lsptr = cur_lop->u1.conval;
        llen = size_of(DDTG(ldtype));
        rsptr = cur_rop->u1.conval;
        rlen = size_of(DDTG(rdtype));
        s = getitem(0, llen + rlen);
        BCOPY(s, stb.n_base + CONVAL1G(lsptr), char, llen);
        BCOPY(s + llen, stb.n_base + CONVAL1G(rsptr), char, rlen);

        c = (CONST *)getitem(4, sizeof(CONST));
        BZERO(c, CONST, 1);
        c->id = AC_CONST;
        c->dtype = get_type(2, TY_CHAR, llen + rlen);
        c->repeatc = 1;
        c->u1.conval = c->sptr = getstring(s, llen + rlen);

        add_to_list(c, &root, &roottail);
        if (--l_repeatc <= 0) {
          cur_lop = cur_lop->next;
          if (cur_lop) {
            r_repeatc = cur_lop->repeatc;
          }
        }
        if (--r_repeatc <= 0) {
          cur_rop = cur_rop->next;
          if (cur_rop) {
            r_repeatc = cur_rop->repeatc;
          }
        }
      }
      return root;
    }
    for (; cur_rop && cur_lop;) {
      c = (CONST *)getitem(4, sizeof(CONST));
      BZERO(c, CONST, 1);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = 1;
      l_conval = cur_lop->u1.conval;
      if (DDTG(cur_lop->dtype) != e_dtype) {
        l_conval = cngcon(l_conval, DDTG(cur_lop->dtype), e_dtype);
      }
      r_conval = cur_rop->u1.conval;
      switch (get_ast_op(op)) {
      case OP_XTOI:
      case OP_XTOK:
      case OP_XTOX:
        /* the front-end sets the correct type for the right operand */
        break;
      default:
        if (DDTG(cur_rop->dtype) != e_dtype) {
          r_conval = cngcon(r_conval, DDTG(cur_rop->dtype), e_dtype);
        }
        break;
      }
      c->u1.conval = init_fold_const(get_ast_op(op), l_conval, r_conval, dt);
      add_to_list(c, &root, &roottail);
      if (--l_repeatc <= 0) {
        cur_lop = cur_lop->next;
        if (cur_lop) {
          l_repeatc = cur_lop->repeatc;
        }
      }
      if (--r_repeatc <= 0) {
        cur_rop = cur_rop->next;
        if (cur_rop) {
          r_repeatc = cur_rop->repeatc;
        }
      }
    }
  } else if (DTY(ldtype) == TY_ARRAY) {
    /* array <binop> scalar */
    cur_lop = lop->id == AC_ACONST ? lop->subc : lop;
    l_repeatc = cur_lop->repeatc;
    e_dtype = DDTG(dtype) != DT_LOG ? DDTG(dtype) : DDTG(lop->dtype);
    r_conval = rop->u1.conval;
    switch (get_ast_op(op)) {
    case OP_XTOI:
    case OP_XTOK:
    case OP_XTOX:
      /* the front-end sets the correct type for the right operand */
      break;
    case OP_CAT:
      rsptr = rop->u1.conval;
      rlen = size_of(DDTG(rdtype));
      for (; cur_lop;) {
        lsptr = cur_lop->u1.conval;
        llen = size_of(DDTG(ldtype));
        s = getitem(0, llen + rlen);
        BCOPY(s, stb.n_base + CONVAL1G(lsptr), char, llen);
        BCOPY(s + llen, stb.n_base + CONVAL1G(rsptr), char, rlen);

        c = (CONST *)getitem(4, sizeof(CONST));
        BZERO(c, CONST, 1);
        c->id = AC_CONST;
        c->dtype = get_type(2, TY_CHAR, llen + rlen);
        c->repeatc = 1;
        c->u1.conval = c->sptr = getstring(s, llen + rlen);

        add_to_list(c, &root, &roottail);
        if (--l_repeatc <= 0) {
          cur_lop = cur_lop->next;
          if (cur_lop) {
            l_repeatc = cur_lop->repeatc;
          }
        }
      }
      return root;
      break;
    default:
      if (rop->dtype != e_dtype) {
        r_conval = cngcon(r_conval, rop->dtype, e_dtype);
      }
    }
    for (; cur_lop;) {
      c = (CONST *)getitem(4, sizeof(CONST));
      BZERO(c, CONST, 1);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = 1;
      l_conval = cur_lop->u1.conval;
      if (DDTG(cur_lop->dtype) != e_dtype) {
        l_conval = cngcon(l_conval, DDTG(cur_lop->dtype), e_dtype);
      }
      mk_cmp(c, op, l_conval, r_conval, rdtype, dt);
      add_to_list(c, &root, &roottail);
      if (--l_repeatc <= 0) {
        cur_lop = cur_lop->next;
        if (cur_lop) {
          l_repeatc = cur_lop->repeatc;
        }
      }
    }
  } else if (DTY(rdtype) == TY_ARRAY) {
    /* scalar <binop> array */
    cur_rop = (rop->id == AC_ACONST ? rop->subc : rop);
    r_repeatc = cur_rop->repeatc;
    e_dtype = DDTG(dtype);
    l_conval = lop->u1.conval;
    if (lop->dtype != e_dtype) {
      l_conval = cngcon(l_conval, lop->dtype, e_dtype);
    }
    if (get_ast_op(op) == OP_CAT) {
      lsptr = lop->u1.conval;
      llen = size_of(DDTG(ldtype));
      for (; cur_rop;) {
        rsptr = cur_rop->u1.conval;
        rlen = size_of(DDTG(rdtype));
        s = getitem(0, llen + rlen);
        BCOPY(s, stb.n_base + CONVAL1G(lsptr), char, llen);
        BCOPY(s + llen, stb.n_base + CONVAL1G(rsptr), char, rlen);

        c = (CONST *)getitem(4, sizeof(CONST));
        BZERO(c, CONST, 1);
        c->id = AC_CONST;
        c->dtype = get_type(2, TY_CHAR, llen + rlen);
        c->repeatc = 1;
        c->u1.conval = c->sptr = getstring(s, llen + rlen);

        add_to_list(c, &root, &roottail);
        if (--r_repeatc <= 0) {
          cur_rop = cur_rop->next;
          if (cur_rop) {
            r_repeatc = cur_rop->repeatc;
          }
        }
      }
      return root;
    }
    for (; cur_rop;) {
      c = (CONST *)getitem(4, sizeof(CONST));
      BZERO(c, CONST, 1);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = 1;
      r_conval = cur_rop->u1.conval;
      switch (get_ast_op(op)) {
      case OP_XTOI:
      case OP_XTOK:
      case OP_XTOX:
        /* the front-end sets the correct type for the right operand */
        break;
      default:
        if (DDTG(cur_rop->dtype) != e_dtype) {
          r_conval = cngcon(r_conval, DDTG(cur_rop->dtype), e_dtype);
        }
      }
      mk_cmp(c, op, l_conval, r_conval, rdtype, dt);
      add_to_list(c, &root, &roottail);
      if (--r_repeatc <= 0) {
        cur_rop = cur_rop->next;
        if (cur_rop) {
          r_repeatc = cur_rop->repeatc;
        }
      }
    }
  } else {
    /* scalar <binop> scalar */
    root = (CONST *)getitem(4, sizeof(CONST));
    BZERO(root, CONST, 1);
    root->id = AC_CONST;
    root->repeatc = 1;
    root->dtype = dt;
    op = get_ast_op(op);
    switch (op) {
    case OP_EQ:
    case OP_GE:
    case OP_GT:
    case OP_LE:
    case OP_LT:
    case OP_NE:
      l_conval =
          init_fold_const(OP_CMP, lop->u1.conval, rop->u1.conval, ldtype);
      switch (op) {
      case OP_EQ:
        l_conval = l_conval == 0;
        break;
      case OP_GE:
        l_conval = l_conval >= 0;
        break;
      case OP_GT:
        l_conval = l_conval > 0;
        break;
      case OP_LE:
        l_conval = l_conval <= 0;
        break;
      case OP_LT:
        l_conval = l_conval < 0;
        break;
      case OP_NE:
        l_conval = l_conval != 0;
        break;
      }
      l_conval = l_conval ? SCFTN_TRUE : SCFTN_FALSE;
      root->u1.conval = l_conval;
      break;
    case OP_LEQV:
      l_conval =
          init_fold_const(OP_CMP, lop->u1.conval, rop->u1.conval, ldtype);
      root->u1.conval = l_conval == 0;
      break;
    case OP_LNEQV:
      l_conval =
          init_fold_const(OP_CMP, lop->u1.conval, rop->u1.conval, ldtype);
      root->u1.conval = l_conval != 0;
      break;
    case OP_LOR:
      root->u1.conval = lop->u1.conval | rop->u1.conval;
      break;
    case OP_LAND:
      root->u1.conval = lop->u1.conval & rop->u1.conval;
      break;
    case OP_XTOI:
    case OP_XTOK:
      root->u1.conval = init_fold_const(op, lop->u1.conval, rop->u1.conval, dt);
      break;
    default:
      l_conval = lop->u1.conval;
      r_conval = rop->u1.conval;
      if (lop->dtype != dt)
        l_conval = cngcon(l_conval, lop->dtype, dt);
      if (rop->dtype != dt)
        r_conval = cngcon(r_conval, rop->dtype, dt);
      root->u1.conval = init_fold_const(op, l_conval, r_conval, dt);
      break;
    }
  }
  return root;
}

static CONST *
eval_array_constructor(CONST *e)
{
  CONST *root = NULL;
  CONST *roottail = NULL;
  CONST *cur_e;
  CONST *new_e;

  /* collapse nested array contstructors */
  for (cur_e = e->subc; cur_e; cur_e = cur_e->next) {
    if (cur_e->id == AC_ACONST) {
      new_e = eval_array_constructor(cur_e);
    } else {
      new_e = eval_init_expr_item(cur_e);
      if (new_e && new_e->id == AC_ACONST) {
        new_e = eval_array_constructor(new_e);
      }
    }
    add_to_list(new_e, &root, &roottail);
  }
  return root;
}

static CONST *
eval_init_expr_item(CONST *cur_e)
{
  CONST *new_e = NULL, *rslt, *rslttail;
  CONST *lop;
  CONST *rop, *temp;
  int repeatc;
  switch (cur_e->id) {
  case AC_IDENT:
    if (STYPEG(cur_e->sptr) == ST_PROC) {
      new_e = clone_init_const(cur_e, true);
      new_e->u1.conval = new_e->sptr;
      new_e->dtype = DINIT_PROC;
      break;
    }
    if (PARAMG(cur_e->sptr) || (DOVARG(cur_e->sptr) && DINITG(cur_e->sptr)) ||
        (CCSYMG(cur_e->sptr) && DINITG(cur_e->sptr))) {
      new_e = clone_init_const_list(
          init_const[PARAMVALG(cur_e->sptr) - 1], true);
      if (cur_e->mbr) {
        new_e->sptr = cur_e->mbr;
      }
    }
    break;
  case AC_CONST:
    new_e = clone_init_const(cur_e, true);
    break;
  case AC_IEXPR:
    if (cur_e->u1.expr.op != AC_INTR_CALL) {
      lop = eval_init_expr(cur_e->u1.expr.lop);
      temp = cur_e->u1.expr.rop;
      if (temp && cur_e->u1.expr.op == AC_ARRAYREF &&
          temp->u1.expr.op == AC_TRIPLE) {
        rop = eval_const_array_triple_section(temp);
      } else
        rop = eval_init_expr(temp);
      new_e = eval_init_op(cur_e->u1.expr.op, lop, cur_e->u1.expr.lop->dtype,
                           rop, rop ? cur_e->u1.expr.rop->dtype : DT_NONE,
                           cur_e->sptr, cur_e->dtype);
    } else {
      new_e = eval_init_op(cur_e->u1.expr.op, cur_e->u1.expr.lop,
                           cur_e->u1.expr.lop->dtype, cur_e->u1.expr.rop,
                           cur_e->u1.expr.rop ? cur_e->u1.expr.rop->dtype : DT_NONE,
                           cur_e->sptr, cur_e->dtype);
    }
    if (cur_e->repeatc > 1) {
      /* need to copy all ict as many times as repeatc*/
      repeatc = cur_e->repeatc;
      rslt = new_e;
      rslttail = new_e;
      while (repeatc > 1) {
        new_e = clone_init_const_list(new_e, true);
        add_to_list(new_e, &rslt, &rslttail);
        --repeatc;
      }
      new_e = rslt;
    }
    new_e->sptr = cur_e->sptr;
    break;
  case AC_ACONST:
    new_e = clone_init_const(cur_e, true);
    new_e->subc = eval_array_constructor(cur_e);
    if (new_e->subc)
      new_e->subc = convert_acl_dtype(new_e->subc, DDTG(new_e->subc->dtype),
                                      DDTG(new_e->dtype));
    break;
  case AC_SCONST:
    new_e = clone_init_const(cur_e, true);
    new_e->subc = eval_init_expr(new_e->subc);
    if (new_e->subc && new_e->subc->dtype == cur_e->dtype) {
      new_e->subc = new_e->subc->subc;
    }
    break;
  case AC_IDO:
    new_e = eval_do(cur_e);
    break;
  }

  return new_e;
}

static CONST *
eval_init_expr(CONST *e)
{
  CONST *root = NULL;
  CONST *roottail = NULL;
  CONST *cur_e;
  CONST *new_e;

  for (cur_e = e; cur_e; cur_e = cur_e->next) {
    switch (cur_e->id) {
    case AC_SCONST:
      new_e = clone_init_const(cur_e, true);
      new_e->subc = eval_init_expr(new_e->subc);
      if (new_e->subc && new_e->subc->dtype == cur_e->dtype) {
        new_e->subc = new_e->subc->subc;
      }
      break;
    case AC_ACONST:
      new_e = clone_init_const(cur_e, true);
      new_e->subc = eval_array_constructor(cur_e);
      if (new_e->subc)
        new_e->subc = convert_acl_dtype(new_e->subc, DDTG(new_e->subc->dtype),
                                        DDTG(new_e->dtype));
      break;
    case AC_IDENT:
      /* need this for AC_MEMBR_SEL */
      if (cur_e->sptr && DTY(DTYPEG(cur_e->sptr)) == TY_ARRAY) {
        new_e = clone_init_const(cur_e, true);
        new_e->subc = eval_init_expr_item(cur_e);
        new_e->sptr = SPTR_NULL;
        new_e->id = AC_ACONST;
        break;
      }
      FLANG_FALLTHROUGH;
    default:
      new_e = eval_init_expr_item(cur_e);
      break;
    }
    add_to_list(new_e, &root, &roottail);
  }

  return root;
}

static CONST *
eval_do(CONST *ido)
{
  ISZ_T i;
  IDOINFO *di = &ido->u1.ido;
  SPTR idx_sptr = di->index_var;
  CONST *idx_ict;
  CONST *root = NULL;
  CONST *roottail = NULL;
  CONST *ict;
  CONST *initict = eval_init_expr_item(di->initval);
  CONST *limitict = eval_init_expr_item(di->limitval);
  CONST *stepict = eval_init_expr_item(di->stepval);
  ISZ_T initval = get_ival(initict->dtype, initict->u1.conval);
  ISZ_T limitval = get_ival(limitict->dtype, limitict->u1.conval);
  ISZ_T stepval = get_ival(stepict->dtype, stepict->u1.conval);
  INT num[2];
  bool inflag = false;

  if (DINITG(idx_sptr) && PARAMVALG(idx_sptr)) {
    idx_ict = init_const[PARAMVALG(idx_sptr) - 1];
  } else {
    idx_ict = (CONST *)getitem(4, sizeof(CONST));
    BZERO(idx_ict, CONST, 1);
    idx_ict->id = AC_CONST;
    idx_ict->dtype = DTYPEG(idx_sptr);
    idx_ict->repeatc = 1;
    save_init(idx_ict, idx_sptr);
    DINITP(idx_sptr, 1); /* MORE use some other flag??? */
  }

  DOVARP(idx_sptr, 1);
  if (stepval >= 0) {
    for (i = initval; i <= limitval; i += stepval) {
      switch (DTY(idx_ict->dtype)) {
      case TY_INT8:
      case TY_LOG8:
        ISZ_2_INT64(i, num);
        idx_ict->u1.conval = getcon(num, idx_ict->dtype);
        break;
      default:
        idx_ict->u1.conval = i;
        break;
      }
      ict = eval_init_expr(ido->subc);
      add_to_list(ict, &root, &roottail);
      inflag = true;
    }
  } else {
    for (i = initval; i >= limitval; i += stepval) {
      switch (DTY(idx_ict->dtype)) {
      case TY_INT8:
      case TY_LOG8:
        ISZ_2_INT64(i, num);
        idx_ict->u1.conval = getcon(num, idx_ict->dtype);
        break;
      default:
        idx_ict->u1.conval = i;
        break;
      }
      ict = eval_init_expr(ido->subc);
      add_to_list(ict, &root, &roottail);
      inflag = true;
    }
  }
  if ((!inflag) && ido->subc) {
    ict = eval_init_expr(ido->subc);
    add_to_list(ict, &root, &roottail);
  }
  DOVARP(idx_sptr, 0);

  return root;
}

static CONST *
clone_init_const(CONST *original, int temp)
{
  CONST *clone;

  if (!original)
    return NULL;
  clone = (CONST *)getitem(4, sizeof(CONST));
  *clone = *original;
  if (clone->subc) {
    clone->subc = clone_init_const_list(original->subc, temp);
  }
  if (clone->id == AC_IEXPR) {
    if (clone->u1.expr.lop) {
      clone->u1.expr.lop = clone_init_const_list(original->u1.expr.lop, temp);
    }
    if (clone->u1.expr.rop) {
      clone->u1.expr.rop = clone_init_const_list(original->u1.expr.rop, temp);
    }
  }
  clone->next = NULL;
  return clone;
}

static CONST *
clone_init_const_list(CONST *original, int temp)
{
  CONST *clone = NULL;
  CONST *clonetail = NULL;

  clone = clone_init_const(original, temp);
  for (original = original->next; original; original = original->next) {
    add_to_list(clone_init_const(original, temp), &clone, &clonetail);
  }
  return clone;
}

static void
add_to_list(CONST *val, CONST **root, CONST **roottail)
{
  CONST *tail;
  if (roottail && *roottail) {
    (*roottail)->next = val;
  } else if (*root) {
    for (tail = *root; tail->next; tail = tail->next)
      ;
    tail->next = val;
  } else {
    *root = val;
  }
  if (roottail && val) { /* find and save the end of the list */
    for (tail = val; tail->next; tail = tail->next)
      ;
    *roottail = tail;
  }
}

static void
save_init(CONST *ict, SPTR sptr)
{
  if (PARAMVALG(sptr)) {
    /* multiple initialization or overlapping initialization error,
     * recognized and reported in assem.c */
    return;
  }

  if (cur_init >= init_list_count) {
    interr("Saved initializer list overflow", init_list_count, ERR_Severe);
    return;
  }
  init_const[cur_init] = ict;
  PARAMVALP(sptr, ++cur_init); /* paramval is cardinal */
}
