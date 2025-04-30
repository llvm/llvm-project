/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
      \file rte.c

        This file contains functions that handle Fortran descriptors including
        array descriptors and type descriptors.
*/

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "machar.h"
#include "ast.h"
#include "semant.h"

#include <stdio.h>

#undef RTE_C
#define RTE_C
#include "rte.h"

static int get_per_dim_member(int, int, int);
static int get_header_member(int sdsc, int info);
#ifdef FLANG_RTE_UNUSED
static int divmod(LOGICAL bIsDiv, int astNum, int astDen, int astRecip,
                  int astShift, int std);
#endif
static char* mangleUnderscores(char* str);
static int lenWithUnderscores(char* str);

static int rte_sc = SC_LOCAL;
static int rte_class = 0;
static int rte_rank = 0;
static int rte_preserve_desc = 0;
static int rte_final_desc = 0;

void
set_descriptor_class(int class)
{
  /* We set rte_class when we want to create a type descriptor in
   * sym_get_sdescr().
   */
  rte_class = class;
}

void
set_descriptor_rank(int r)
{
  /* We set rte_rank when we want to create a descriptor for a pointer or
   * allocatable with room for its dynamic type. Used in sym_get_sdescr().
   */
  rte_rank = r;
}

void
set_preserve_descriptor(int d)
{
  /* We set rte_preserve_desc when we do not want to override an
   * existing descriptor for a particular symbol in sym_get_sdescr().
   * This typically occurs with descriptors associated with polymorphic
   * objects.
   */
  rte_preserve_desc = d;
}

void
set_final_descriptor(int d)
{
  /* We set this when we want to general a final procedure descriptor */
  rte_final_desc = d;
}

void
set_descriptor_sc(int sc)
{
  rte_sc = sc;
  set_symutl_sc(sc);
}

int
get_descriptor_sc(void)
{
  return rte_sc;
}

/* \brief Returns a length of a string in which we add 1 to length for each
 * underscore in the string.
 *
 * This function is used in sym_get_sdescr() to mangle a type descriptor
 * name. See also mangleUnderscores() below.
 *
 * \param str is the string we're processing.
 *
 * \returns length of str plus number of underscores.
 */
static int
lenWithUnderscores(char* str)
{
  int len;

  if (str == NULL)
    return 0;

  for(len=0; *str != '\0'; ++len, ++str) {
    if (*str == '_') {
      ++len;
    }
  }
  return len;
} 

/* \brief If a string has underscores, then we append an equal number of $ 
 * to the string.
 * 
 * This function is called by sym_get_sdescr(). It is used in the
 * construction of type descriptor and final descriptor symbol names.
 *
 * Internally, $ is used as part of a suffix to a symbol name. Distinguishing
 * between the prefix and suffix is required later, so we need to use $ and
 * not underscores here. The $ are changed to underscores just before the
 * symbol is written out in the backend.
 *
 * Appending extra $ prevents name conflicts between type descriptor
 * objects that have an underscore in their type name, host module name,
 * and/or subprogram name. 
 *
 * \param str is the string we are processing.
 *
 * \returns str if NULL or if no underscores are present. Otherwise, returns a
 * mangled name.
 */
static char*
mangleUnderscores(char* str)
{
  char * newStr;
  int i, len, lenscores;
    
  if (str == NULL || strchr(str, '_') == 0)
    return str;
  
  lenscores = lenWithUnderscores(str);
  newStr = getitem(0, lenscores+1);
  len = strlen(str);
  strcpy(newStr, str);
  for(i=len; i < lenscores; ++i) {
    newStr[i] = '$';
  }
  newStr[i] = '\0';
  return newStr;
}  

/* \brief Generate an array section descriptor, type descriptor, or final
 * descriptor.
 *
 * \param sptr is the symbol table pointer receiving the descriptor
 * 
 * \param is the rank for an array section descriptor.
 *
 * \returns the descriptor symbol table pointer.
 */
int
sym_get_sdescr(int sptr, int rank)
{
  int dtype;
  int ub;
  int sdsc, sdsc_mem;
  LOGICAL addit = FALSE;

  if (SDSCG(sptr) > NOSYM && rte_preserve_desc) {
    /* We must preserve descriptors associated with polymorphic objects */
    return SDSCG(sptr);
  }
  if (rank < 0) {
    dtype = DTYPEG(sptr);
    if (DTY(dtype) != TY_ARRAY) {
      rank = 0;
    } else {
      rank = ADD_NUMDIM(dtype);
      if (STYPEG(sptr) == ST_MEMBER &&
          (ALIGNG(sptr) || DISTG(sptr) || POINTERG(sptr) || ADD_DEFER(dtype) ||
           ADD_ADJARR(dtype) || ADD_NOBOUNDS(dtype))) {
        /* section descriptor must be added to derived type */
        addit = TRUE;
      }
    }
  }
  if ((CLASSG(sptr) || FINALIZEDG(sptr) /*|| ALLOCDESCG(sptr)*/) &&
      STYPEG(sptr) == ST_MEMBER && rte_rank) {
    addit = TRUE;
  }
  if (rank || rte_rank) {
    ub = DESC_HDR_LEN + rank * DESC_DIM_LEN;
  } else if (rank == 0 &&
             (DTY(DTYPEG(sptr)) == TY_CHAR || DTY(DTYPEG(sptr)) == TY_NCHAR)) {
    /* Do we really need 18 of them or just F90_Desc? */
    ub = DESC_HDR_LEN + 1 * DESC_DIM_LEN;
    if (STYPEG(sptr) == ST_MEMBER)
      addit = TRUE;
  } else if (DTY(DTYPEG(sptr)) == TY_PTR || IS_PROC_DUMMYG(sptr)) {
    /* special descriptor for procedure pointer */
    ub = DESC_HDR_LEN;
    if (STYPEG(sptr) == ST_MEMBER)
      addit = TRUE;
  } else {
    ub = 1;
  }

  dtype = get_array_dtype(1, astb.bnd.dtype);
  ADD_LWBD(dtype, 0) = 0;
  ADD_LWAST(dtype, 0) = astb.bnd.one;
  ADD_NUMELM(dtype) = ADD_UPBD(dtype, 0) = ADD_UPAST(dtype, 0) =
      mk_isz_cval(ub, astb.bnd.dtype);
  if (rte_class) {
    /* create type descriptor */
    int tag;
    int sc;

    assert(DTY(DTYPEG(sptr)) == TY_DERIVED,
           "sym_get_sdescr: making type descriptor for non-derived type", sptr,
           3);
    tag = DTY(DTYPEG(sptr) + 3);

    if (SDSCG(tag) && !rte_final_desc) {
      sdsc = SDSCG(tag);
      sc = SCG(sdsc);
    } else {
      char *desc_sym;
      int len = lenWithUnderscores(SYMNAME(gbl.currmod)) + 
                lenWithUnderscores(SYMNAME(SCOPEG(tag))) +
                lenWithUnderscores(SYMNAME(SCOPEG(gbl.currsub))) +
                lenWithUnderscores(SYMNAME(tag)) + 3 /* 3 for "$$\0" */; 
      desc_sym = getitem(0, len);
      if (strcmp(SYMNAME(gbl.currmod), SYMNAME(SCOPEG(tag))) == 0) {
        sprintf(desc_sym, "%s$%s",mangleUnderscores(SYMNAME(gbl.currmod)), 
                mangleUnderscores(SYMNAME(tag)));
        sc = SC_EXTERN;
      } else {
        if (gbl.currmod) {
          sprintf(desc_sym, "%s$%s$%s", mangleUnderscores(SYMNAME(gbl.currmod)),
                  mangleUnderscores(SYMNAME(SCOPEG(tag))), 
                  mangleUnderscores(SYMNAME(tag)));
          sc = SC_EXTERN;
        } else if (gbl.currsub) {
          sprintf(desc_sym, "%s$%s", mangleUnderscores(SYMNAME(gbl.currsub)), 
                  mangleUnderscores(SYMNAME(tag)));
          sc = SC_STATIC;
        } else {
          sprintf(desc_sym, "%s$%s", mangleUnderscores(SYMNAME(SCOPEG(tag))), 
                  mangleUnderscores(SYMNAME(tag)));
          sc = SC_STATIC;
        }
      }
      if (rte_final_desc) {
        /* create a special "final" descriptor used to store
         * final procedures of a type.
         */
        assert((strlen(desc_sym)+7) <= (MAXIDLEN+1), 
               "sym_get_sdescr: final desc name buffer overflow", MAXIDLEN, 4);
        sdsc = getsymf("%s$td$ft", desc_sym);
        HCCSYMP(sdsc, 1);
        HIDDENP(sdsc, 1); /* can't see this, if in the parser */
        SCOPEP(sdsc, stb.curr_scope);
        if (gbl.internal > 1)
          INTERNALP(sdsc, 1);
        FINALP(sdsc, 1);
        PARENTP(sdsc, DTYPEG(sptr));
      } else {
        assert((strlen(desc_sym)+3) <= (MAXIDLEN+1), 
               "sym_get_sdescr: type desc name buffer overflow", MAXIDLEN, 4);
        sdsc = get_next_sym(desc_sym, "td");
        SDSCP(tag, sdsc);
        if (get_struct_initialization_tree(DTYPEG(sptr))) {
          /* Ensure existence of template object if there's initializers. */
          (void) get_dtype_init_template(DTYPEG(sptr));
        }
      }
    }

    CLASSP(sdsc, 1);

    UNLPOLYP(sdsc, UNLPOLYG(tag));
    DTYPEP(sdsc, dtype);
    STYPEP(sdsc, ST_DESCRIPTOR);
    DCLDP(sdsc, 1);

    SCP(sdsc, sc);

    NODESCP(sdsc, 1);
    DESCARRAYP(sdsc, 1); /* used in detect.c */
    if (INTERNALG(sptr))
      INTERNALP(sdsc, 1);
    if (CONSTRUCTSYMG(sptr)) {
      CONSTRUCTSYMP(sdsc, true);
      ENCLFUNCP(sdsc, ENCLFUNCG(sptr));
    }
    return sdsc;
  }
  sdsc_mem = 0;
  if (rte_rank && SDSCG(sptr)) {
    /* This occurs when we're passing in a non-polymorphic alloc/ptr object
     * to a polymorphic dummy argument. We want to preserve the descriptor
     * of the actual since it may contain associate, etc. info but we also
     * need to enlarge it to store its type in it. See the call to
     * get_static_descriptor() in check_alloc_ptr_type().
     */
    sdsc = SDSCG(sptr);
  } else {
    if (addit && !rte_rank) {
      sdsc = get_next_sym_dt(SYMNAME(sptr), "sd", ENCLDTYPEG(sptr));
    } else {
      sdsc = get_next_sym(SYMNAME(sptr), "sd");
    }
  }
  if (addit && rte_rank) {
    /* Create a type descriptor that will be added to derived type */
    sdsc_mem = get_next_sym_dt(SYMNAME(sptr), "td", ENCLDTYPEG(sptr));
    DTYPEP(sdsc_mem, dtype);
    STYPEP(sdsc_mem, ST_DESCRIPTOR);
    DCLDP(sdsc_mem, 1);
    SCP(sdsc_mem, rte_sc);
    NODESCP(sdsc_mem, 1);
    DESCARRAYP(sdsc_mem, 1); /* used in detect.c */
    CLASSP(sdsc_mem, 1);
    if (INTERNALG(sptr))
      INTERNALP(sdsc_mem, 1);
    if (rte_sc == SC_PRIVATE && ALLOCATTRG(sptr) && MIDNUMG(sptr) &&
        !SDSCG(sptr)) {
      if (SCG(MIDNUMG(sptr)) != SC_PRIVATE) {
        SCP(sdsc_mem, SC_LOCAL);
      }
    }
  }
  DTYPEP(sdsc, dtype);
  STYPEP(sdsc, ST_DESCRIPTOR);
  DCLDP(sdsc, 1);
  SCP(sdsc, rte_sc);
  NODESCP(sdsc, 1);
  DESCARRAYP(sdsc, 1); /* used in detect.c */
  if (DTY(DTYPEG(sptr)) == TY_PTR || IS_PROC_DUMMYG(sptr)) {
    IS_PROC_DESCRP(sdsc, 1);
  }
  if (INTERNALG(sptr))
    INTERNALP(sdsc, 1);
  if (rte_sc == SC_PRIVATE && ALLOCATTRG(sptr) && MIDNUMG(sptr) &&
      !SDSCG(sptr)) {
    if (SCG(MIDNUMG(sptr)) != SC_PRIVATE) {
      SCP(sdsc, SC_LOCAL);
    }
  }
#ifdef DEVICEG
  if (CUDAG(gbl.currsub) & (CUDA_DEVICE | CUDA_GLOBAL)) {
    /* copy the device bit to the descriptor */
    if (DEVICEG(sptr))
      DEVICEP(sdsc, 1);
  }
#endif

  if (sdsc_mem || addit) {
    /* Need to add type or static descriptor to data type after sptr */
    int dtype = ENCLDTYPEG(sptr);
    assert(STYPEG(sptr) == ST_MEMBER, "sym_get_sdescr: sptr must be member",
           sptr, 3);
    assert(dtype && DTY(dtype) == TY_DERIVED,
           "sym_get_sdescr: sptr is member without enclosing dtype", sptr, 3);
    if (dtype && DTY(dtype) == TY_DERIVED) {
      int mem;
      for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
        if (mem == sptr) {
          /* add descriptor to the datatype just after 'sptr' */
          int new_mem = sdsc_mem ? sdsc_mem : sdsc;
          STYPEP(new_mem, ST_MEMBER);
          ENCLDTYPEP(new_mem, dtype);
          SYMLKP(new_mem, SYMLKG(sptr));
          if (SYMLKG(sptr) > NOSYM)
            VARIANTP(SYMLKG(sptr), sdsc_mem); /* previous link */
          SYMLKP(sptr, new_mem);
          VARIANTP(new_mem, sptr);
          SCP(new_mem, SC_NONE);
          break;
        }
      }
      assert(mem == sptr,
             "sym_get_sdescr: sptr not member of its enclosing derived type",
             sptr, 3);
    }
  }
  if (CONSTRUCTSYMG(sptr)) {
    CONSTRUCTSYMP(sdsc, true);
    ENCLFUNCP(sdsc, ENCLFUNCG(sptr));
  }
  return sdsc;
} /* sym_get_sdescr */

int
sym_get_place_holder(char *basename, int dtype)
{
  int sptr;

  sptr = sym_get_scalar(basename, "pholder", dtype);
  return sptr;
}

void
get_static_descriptor(int sptr)
{
  int sdsc;

  sdsc = sym_get_sdescr(sptr, -1);
  SDSCP(sptr, sdsc);

  if (!is_procedure_ptr(sptr) && !IS_PROC_DUMMYG(sptr)) {
    NOMDCOMP(sdsc, 1);
    LNRZDP(sptr, 1);
  }
}

/*   sptr		   - base array
 *   MIDNUMG(sptr)	   - pointer to base array
 *   PTROFFG(sptr)	   - offset to base array  (optional)
 *   SECDSCG(DESCRG(sptr)) - base descriptor (desc)
 *   [ desc = SECDSCG(DESCRG(sptr)) ]
 *   MIDNUMG(desc)	   - pointer to descriptor
 *   PTROFFG(desc)	   - offset to descriptor
 */
void
get_all_descriptors(int sptr)
{
  int pvar;
  int ovar;
  int arrdsc;
  int ndim;
  int ast;
  int i;
  ADSC *ad;

  /*
   * All associated variables created for the pointer object cannot appear
   * in the module common if the object is in the specification part of a
   * module.	A pointer common block will be created for the object.
   * If in a module, this common block is created by the module processing
   * and its member list contains these variables.  For a local object,
   * the pointer common block is created by astout by just emitting the
   * appropriate common statement.  To prevent the variables from appearing
   * in the module common, set their NOMDCOM ('not in module common') flags.
   */

  /* create pointer or use one already created if F77OUTPUT */
  pvar = MIDNUMG(sptr);
  if (pvar == 0 || pvar == NOSYM || CCSYMG(pvar)) {
    pvar = sym_get_ptr(sptr);
    MIDNUMP(sptr, pvar);
  }
  NOMDCOMP(pvar, 1);

  /* create offset */
  ovar = sym_get_offset(sptr);
  PTROFFP(sptr, ovar);
  NOMDCOMP(ovar, 1);

  ndim = rank_of_sym(sptr);

  if (STYPEG(sptr) == ST_MEMBER && SYMLKG(sptr) != pvar) {
    int dtype;
    dtype = ENCLDTYPEG(sptr);
    if (dtype) {
      int mem;
      for (mem = DTY(dtype + 1); mem > NOSYM && mem != sptr; mem = SYMLKG(mem))
        ;
      if (mem == sptr) {
        /* add 'pointer' to the derived type just after 'sptr' */
        int next = SYMLKG(sptr);
        SYMLKP(sptr, pvar);
        STYPEP(pvar, ST_MEMBER);
        ENCLDTYPEP(pvar, dtype);
        SYMLKP(pvar, next);
        VARIANTP(pvar, sptr);
        SCP(pvar, SC_NONE);
        if (next > NOSYM)
          VARIANTP(next, pvar);
        if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
          /* for arrays, put 'offset' there too */
          SYMLKP(pvar, ovar);
          STYPEP(ovar, ST_MEMBER);
          ENCLDTYPEP(ovar, dtype);
          SYMLKP(ovar, next);
          VARIANTP(ovar, pvar);
          SCP(ovar, SC_NONE);
          if (next > NOSYM)
            VARIANTP(next, ovar);
        }
      }
    }
  }
  if (DTY(DTYPEG(sptr)) != TY_ARRAY)
    return;

  /* create descriptor and pointer */
  trans_mkdescr(sptr);
  arrdsc = DESCRG(sptr);
  assert(SDSCG(sptr), "get_all_descriptors, need static descriptor", sptr, 2);
  SECDSCP(arrdsc, SDSCG(sptr));

  if (STYPEG(sptr) == ST_MEMBER) {
    /* don't change bounds of distributed member arrays unless
     * allocatable or pointer */
    if (!POINTERG(sptr) && !ALLOCG(sptr)) {
      return;
    }
  }

  /*
   * for the descriptor (whose storage class is SC_BASED), set its
   * 'not in module common' flag so that the module and interf processing
   * will ignore this 'based' symbol.  The module processing creates the
   * pointer common block early, but if pointers aren't allowed in the output,
   * the descriptor needs to be added to the beginning of the common.  The
   * descriptor cannot be added to the common by the module processing, since
   * its storage class is SC_BASED; astout will need to write the necessary
   * common statement.
   */

  /* change the bounds of the array and its descriptor */
  ad = AD_DPTR(DTYPEG(sptr));
  DESCUSEDP(sptr, 1);
  ast_visit(1, 1);
  for (i = 0; i < ndim; i++) {
    /* TBD it would be nice to zap the z_b... variables so their decls
     * do not appear in the output.
     */
    int oldast, a;

    oldast = AD_LWAST(ad, i);
    AD_LWAST(ad, i) = get_global_lower(SDSCG(sptr), i);
    if (oldast)
      ast_replace(oldast, AD_LWAST(ad, i));

    oldast = AD_UPAST(ad, i);
    a = get_extent(SDSCG(sptr), i);
    a = mk_binop(OP_SUB, a, mk_isz_cval(1, astb.bnd.dtype), astb.bnd.dtype);
    a = mk_binop(OP_ADD, AD_LWAST(ad, i), a, astb.bnd.dtype);
    AD_UPAST(ad, i) = a;
    if (oldast)
      ast_replace(oldast, AD_UPAST(ad, i));

    oldast = AD_EXTNTAST(ad, i);
    AD_EXTNTAST(ad, i) = get_extent(SDSCG(sptr), i);
    if (oldast)
      ast_replace(oldast, AD_EXTNTAST(ad, i));

    {
      AD_LWBD(ad, i) = AD_LWAST(ad, i);
      AD_UPBD(ad, i) = AD_UPAST(ad, i);
    }
  }
  for (i = 0; i < ndim; ++i) {
    AD_MLPYR(ad, i) = get_local_multiplier(SDSCG(sptr), i);
  }
  AD_NUMELM(ad) = get_desc_gsize(SDSCG(sptr));
  AD_ZBASE(ad) = get_xbase(SDSCG(sptr));
  ast = AD_ZBASE(ad);
  if (ast)
    AD_ZBASE(ad) = ast_rewrite(ast);
  ast_unvisit();
}

int
get_multiplier_index(int dim)
{
  return DESC_HDR_LEN + dim * DESC_DIM_LEN + DESC_DIM_LMULT;
}

int
get_global_lower_index(int dim)
{
  return DESC_HDR_LEN + dim * DESC_DIM_LEN + DESC_DIM_LOWER;
}

int
get_global_upper_index(int dim)
{
  return DESC_HDR_LEN + dim * DESC_DIM_LEN + DESC_DIM_UPPER;
}

int
get_global_extent_index(int dim)
{
  return DESC_HDR_LEN + dim * DESC_DIM_LEN + DESC_DIM_EXTENT;
}

int
get_section_stride_index(int dim)
{
  return DESC_HDR_LEN + dim * DESC_DIM_LEN + DESC_DIM_SSTRIDE;
}

int
get_section_offset_index(int dim)
{
  return DESC_HDR_LEN + dim * DESC_DIM_LEN + DESC_DIM_SOFFSET;
}

int
get_xbase_index(void)
{
  return DESC_HDR_LBASE;
}

int
get_xbase(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_LBASE);
}

int
get_gbase(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_GBASE);
}

int
get_gbase2(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_GBASE + 1);
}

int
get_kind(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_KIND);
}

int
get_byte_len(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_BYTE_LEN);
}

int
get_byte_len_indx(void)
{
  return DESC_HDR_BYTE_LEN;
}

int
get_local_multiplier(int sdsc, int dim)
{
  return get_per_dim_member(sdsc, dim, DESC_DIM_LMULT);
}

int
get_global_lower(int sdsc, int dim)
{
  return get_per_dim_member(sdsc, dim, DESC_DIM_LOWER);
}

int
get_global_upper(int sdsc, int dim)
{
  return get_per_dim_member(sdsc, dim, DESC_DIM_UPPER);
}

int
get_extent(int sdsc, int dim)
{
  return get_per_dim_member(sdsc, dim, DESC_DIM_EXTENT);
}

int
get_section_stride(int sym, int dim)
{
  return get_per_dim_member(sym, dim, DESC_DIM_SSTRIDE);
}

int
get_section_offset(int sym, int dim)
{
  return get_per_dim_member(sym, dim, DESC_DIM_SOFFSET);
}

int
get_local_lower(int sdsc, int dim)
{
  return get_global_lower(sdsc, dim);
}

int
get_local_upper(int sdsc, int dim)
{
  return get_global_upper(sdsc, dim);
}

int
get_owner_lower(int sdsc, int dim)
{
  return get_global_lower(sdsc, dim);
}

int
get_owner_upper(int sdsc, int dim)
{
  return get_global_upper(sdsc, dim);
}

int
get_heap_block(int sdsc)
{
  return astb.i0;
}

int
get_desc_rank(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_RANK);
}

int
get_desc_tag(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_TAG);
}

int
get_desc_flags(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_FLAGS);
}

int
get_desc_gsize_index(void)
{
  return DESC_HDR_GSIZE;
}

int
get_desc_gsize(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_GSIZE);
}

int
get_desc_lsize(int sdsc)
{
  return get_header_member(sdsc, DESC_HDR_LSIZE);
}

int
get_lsize_index(void)
{
  return DESC_HDR_LSIZE;
}

int
get_proc_base(int sdsc)
{
  return astb.i0;
}

int
get_proc_shape(int sdsc, int dim)
{
  return astb.i1;
}

int
get_proc_stride(int sdsc, int dim)
{
  return get_section_stride(sdsc, dim);
}

int
get_block_size(int sdsc, int dim)
{
  return astb.i0;
}

int
get_neg_ovlp(int sdsc, int dim)
{
  return astb.i0;
}

int
get_pos_ovlp(int sdsc, int dim)
{
  return astb.i0;
}

int
get_template_offset(int sdsc, int dim)
{
  return get_section_offset(sdsc, dim);
}

int
get_descriptor_len(int rank)
{
  return DESC_HDR_LEN + rank * DESC_DIM_LEN;
}

static int
get_header_member(int sdsc, int info)
{
  int ast;
  int subs[1];

#if DEBUG
  if (!sdsc)
    interr("get_header_member, blank static descriptor", 0, 3);
  else if (STYPEG(sdsc) != ST_ARRDSC && STYPEG(sdsc) != ST_DESCRIPTOR &&
           DTY(DTYPEG(sdsc)) != TY_ARRAY)
    interr("get_header_member, bad static descriptor", sdsc, 3);
#endif
  subs[0] = mk_isz_cval(info, astb.bnd.dtype);
  ast = mk_subscr(mk_id(sdsc), subs, 1, astb.bnd.dtype);
  return ast;
}


/** \brief Generate an AST for accessing a particular field in a descriptor
 *         header.
 *
 * Note: This is similar to get_header_member() above except it also 
 * operates on descriptors that are embedded in derived type objects.
 *
 * \param parent is the ast of the expression with the descriptor that
 *        we want to access. This is needed if the descriptor is embedded
 *        in a derived type object.
 * \param sdsc is the symbol table pointer of the descriptor we want to
 *        access.
 * \param info is the field we want to access in the descriptor.
 *
 * \return an ast expression of the descriptor access.
 */
int
get_header_member_with_parent(int parent, int sdsc, int info)
{
  int ast;
  int subs[1];

#if DEBUG
  if (!sdsc)
    interr("get_header_member, blank static descriptor", 0, 3);
  else if (STYPEG(sdsc) != ST_ARRDSC && STYPEG(sdsc) != ST_DESCRIPTOR &&
           DTY(DTYPEG(sdsc)) != TY_ARRAY)
    interr("get_header_member, bad static descriptor", sdsc, 3);
#endif
  subs[0] = mk_isz_cval(info, astb.bnd.dtype);
  ast = mk_subscr(check_member(parent, mk_id(sdsc)), subs, 1, astb.bnd.dtype);
  return ast;
}

#ifdef FLANG_RTE_UNUSED
static int
get_array_rank(int sdsc)
{
  int rank = 0;

  if (STYPEG(sdsc) == ST_ARRDSC) {
    rank = rank_of_sym(ARRAYG(sdsc));
  } else if (STYPEG(sdsc) == ST_DESCRIPTOR || STYPEG(sdsc) == ST_MEMBER) {
    int dtype = DTYPEG(sdsc);
    int ubast = AD_UPAST(AD_DPTR(dtype), 0);
    int ub;
    assert(DTY(dtype) == TY_ARRAY && A_TYPEG(ubast) == A_CNST,
           "get_array_rank: Invalid ST_DESCRIPTOR|ST_MEMBER dtype", DTY(dtype),
           0);
    ub = CONVAL2G(A_SPTRG(ubast));

    rank = (ub - (DESC_HDR_LEN + HPF_DESC_HDR_LEN)) /
           (DESC_DIM_LEN + HPF_DESC_DIM_LEN);
  } else {
    assert(0, "get_array_rank: Invalid descriptor type", STYPEG(sdsc), 0);
  }

  return rank;
}
#endif

static int
get_per_dim_member(int sdsc, int dim, int info)
{
  int ast;
  int subs[1];

#if DEBUG
  assert(sdsc && (STYPEG(sdsc) == ST_DESCRIPTOR || STYPEG(sdsc) == ST_ARRDSC ||
                  STYPEG(sdsc) == ST_MEMBER),
         "get_per_dim_member-illegal stat.desc", sdsc, 0);
#endif
  subs[0] =
      mk_isz_cval(DESC_HDR_LEN + dim * DESC_DIM_LEN + info, astb.bnd.dtype);
  ast = mk_subscr(mk_id(sdsc), subs, 1, astb.bnd.dtype);
  return ast;
}

#ifdef FLANG_RTE_UNUSED
/* If bIsDiv is TRUE, add statements to compute astNum/astDen before std.
 * If bIsDiv is FALSE, add statements to compute astNum%astDen before std.
 * astRecip is the reciprocal of astDen. If astShift is nonzero,
 * 2**astShift = astDen, and can be used to shift astNum.
 * Return an AST representing the result. */
static int
divmod(LOGICAL bIsDiv, int astNum, int astDen, int astRecip, int astShift,
       int std)
{
  int sptr;
  int astRes;
  int ast, ast1, astStmt;

  if (astDen == astb.i1)
    return astNum;
  if (astNum == astb.i0)
    return astNum;

  sptr = sym_get_scalar("r", "rte", DT_INT);
  astRes = mk_id(sptr);

  if (!XBIT(49, 0x01000000)) {
    /* ...not T3D/T3E target. */
    astStmt = mk_stmt(A_IFTHEN, 0);
    ast = mk_binop(OP_LT, astShift, astb.i0, DT_LOG);
    A_IFEXPRP(astStmt, ast);
    add_stmt_before(astStmt, std);

    ast = mk_binop(OP_DIV, astNum, astDen, DT_INT);
    if (!bIsDiv) {
      ast = mk_binop(OP_MUL, astDen, ast, DT_INT);
      ast = mk_binop(OP_SUB, astNum, ast, DT_INT);
    }
    astStmt = mk_assn_stmt(astRes, ast, DT_INT);
    add_stmt_before(astStmt, std);

    astStmt = mk_stmt(A_ELSE, 0);
    add_stmt_before(astStmt, std);

    ast1 = mk_unop(OP_SUB, astShift, DT_INT);
    ast = ast_intr(I_ISHFT, DT_INT, 2, astNum, ast1);
    if (!bIsDiv) {
      ast = ast_intr(I_ISHFT, DT_INT, 2, ast, astShift);
      ast = ast_intr(I_IEOR, DT_INT, 2, astNum, ast);
    }
    astStmt = mk_assn_stmt(astRes, ast, DT_INT);
    add_stmt_before(astStmt, std);

    astStmt = mk_stmt(A_ENDIF, 0);
    add_stmt_before(astStmt, std);
    return astRes;
  }
  astStmt = mk_stmt(A_IFTHEN, 0);
  ast = mk_binop(OP_EQ, astRecip, astb.i0, DT_LOG);
  A_IFEXPRP(astStmt, ast);
  add_stmt_before(astStmt, std);

  /* Denominator is 1. */
  if (bIsDiv)
    astStmt = mk_assn_stmt(astRes, astNum, DT_INT);
  else
    astStmt = mk_assn_stmt(astRes, astb.i0, DT_INT);
  add_stmt_before(astStmt, std);

  astStmt = mk_stmt(A_ELSE, 0);
  add_stmt_before(astStmt, std);

  sptr = sym_mkfunc_nodesc("int_mult_upper", DT_INT);
  ast = begin_call(A_FUNC, sptr, 2);
  add_arg(mk_default_int(astNum));
  add_arg(mk_default_int(astRecip));

  if (!bIsDiv) {
    ast = mk_binop(OP_MUL, astDen, ast, DT_INT);
    ast = mk_binop(OP_SUB, astNum, ast, DT_INT);
  }
  astStmt = mk_assn_stmt(astRes, ast, DT_INT);
  add_stmt_before(astStmt, std);

  astStmt = mk_stmt(A_ENDIF, 0);
  add_stmt_before(astStmt, std);

  return astRes;
}
#endif
