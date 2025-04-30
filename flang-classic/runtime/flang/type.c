/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief F2003 polymorphic/OOP runtime support
 */

#include "type.h"
#include "f90alloc.h"
#include "stdioInterf.h"

static struct type_desc *I8(__f03_ty_to_id)[];

void ENTF90(SET_INTRIN_TYPE, set_intrin_type)(F90_Desc *dd,
                                              __INT_T intrin_type);

static TYPE_DESC *get_parent_pointer(TYPE_DESC *src_td, __INT_T level);

static void sourced_alloc_and_assign_array(int extent, char *ab, char *bb,
                                           TYPE_DESC *td);
static void sourced_alloc_and_assign_array_from_scalar(int extent, char *ab,
                                                       char *bb, TYPE_DESC *td);

static void get_source_and_dest_sizes(F90_Desc *ad, F90_Desc *bd, int *dest_sz,
                                      int *src_sz, int *dest_is_array,
                                      int *src_is_array, TYPE_DESC **tad,
                                      TYPE_DESC **tbd, __INT_T flag);
static int has_intrin_type(F90_Desc *dd);

#define ARG1_PTR 0x1
#define ARG1_ALLOC 0x2
#define ARG2_PTR 0x4
#define ARG2_ALLOC 0x8
#define ARG2_INTRIN 0x10

__LOG_T
ENTF90(SAME_TYPE_AS, same_type_as)(void *ab, OBJECT_DESC *ad, void *bb,
                                   OBJECT_DESC *bd, __INT_T flag, ...)
{
  OBJECT_DESC *t1 = ad, *t2 = bd;
  TYPE_DESC *atd, *btd;

  if (!ad || !bd)
    return 0;

  if (flag) {
    va_list va;
    int is_unl_poly = 0;
    va_start(va, flag);
    if (flag & (ARG1_PTR | ARG1_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG1_PTR) &&
            ENTFTN(ASSOCIATED, associated)(ab, (F90_Desc *)ad, 0, 0)) &&
          !I8(__fort_allocated)(ab)) {
        t1 = vatd;
        is_unl_poly |= t1->tag == __POLY && t1->baseTag == __POLY;
      }
    }
    if (flag & (ARG2_PTR | ARG2_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG2_PTR) &&
            ENTFTN(ASSOCIATED, associated)(bb, (F90_Desc *)bd, 0, 0)) &&
          !I8(__fort_allocated)(bb)) {
        t2 = vatd;
        is_unl_poly |= t2->tag == __POLY && t2->baseTag == __POLY;
      }
    }
    va_end(va);
    if (is_unl_poly)
      return 0;
  }

  atd = t1->type ? t1->type : (TYPE_DESC *)t1;
  btd = t2->type ? t2->type : (TYPE_DESC *)t2;
  return atd == btd ? GET_DIST_TRUE_LOG : 0;
}

__LOG_T
ENTF90(EXTENDS_TYPE_OF, extends_type_of)(void *ab, OBJECT_DESC *ad, void *bb,
                                         OBJECT_DESC *bd, __INT_T flag, ...)
{
  OBJECT_DESC *t1 = ad, *t2 = bd;
  TYPE_DESC *atd, *btd;

  if (!ad || !bd)
    return 0;

  if (flag) {
    va_list va;
    int is_unl_poly_arg1 = 0, is_unl_poly_arg2 = 0;

    va_start(va, flag);
    if (flag & (ARG1_PTR | ARG1_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG1_PTR) &&
            ENTFTN(ASSOCIATED, associated)(ab, (F90_Desc *)ad, 0, 0)) &&
          !I8(__fort_allocated)(ab)) {
        t1 = vatd;
        is_unl_poly_arg1 = t1->tag == __POLY && t1->baseTag == __POLY;
      }
    }
    if (flag & (ARG2_PTR | ARG2_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG2_PTR) &&
            ENTFTN(ASSOCIATED, associated)(bb, (F90_Desc *)bd, 0, 0)) &&
          !I8(__fort_allocated)(bb)) {
        t2 = vatd;
        is_unl_poly_arg2 = t2->tag == __POLY && t2->baseTag == __POLY;
      }
    }
    va_end(va);

    if (is_unl_poly_arg2) {
      /* if second argument is unlimited polymorphic and it's
       * disassociated pointer or unallocated allocatable, then return TRUE.
       */
      return GET_DIST_TRUE_LOG;
    }
    if (is_unl_poly_arg1) {
      /* if first argument is unlimited polymorphic and it's either a
       * disassociated pointer or unallocated allocatable, then return FALSE.
       */
      return 0;
    }
  }

  atd = (t1->type /*&& t1->tag > 0 && t1->tag < __NTYPES*/) ? t1->type
                                                            : (TYPE_DESC *)t1;
  btd = (t2->type /*&& t2->tag > 0 && t2->tag < __NTYPES*/) ? t2->type
                                                            : (TYPE_DESC *)t2;
  if (atd == btd)
    return GET_DIST_TRUE_LOG;

  if (atd->obj.level > btd->obj.level) {
    TYPE_DESC *parent = get_parent_pointer(atd, btd->obj.level + 1);
    if (btd == parent)
      return GET_DIST_TRUE_LOG;
  }

  return 0;
}

/* Identical to same_type_as() above apart from name and result type. */
__LOG8_T
ENTF90(KSAME_TYPE_AS, ksame_type_as)(void *ab, OBJECT_DESC *ad, void *bb,
                                     OBJECT_DESC *bd, __INT_T flag, ...)
{
  OBJECT_DESC *t1 = ad, *t2 = bd;
  TYPE_DESC *atd, *btd;

  if (!ad || !bd)
    return 0;

  if (flag) {
    va_list va;
    int is_unl_poly = 0;
    va_start(va, flag);
    if (flag & (ARG1_PTR | ARG1_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG1_PTR) &&
            ENTFTN(ASSOCIATED, associated)(ab, (F90_Desc *)ad, 0, 0)) &&
          !I8(__fort_allocated)(ab)) {
        t1 = vatd;
        is_unl_poly |= t1->tag == __POLY && t1->baseTag == __POLY;
      }
    }
    if (flag & (ARG2_PTR | ARG2_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG2_PTR) &&
            ENTFTN(ASSOCIATED, associated)(bb, (F90_Desc *)bd, 0, 0)) &&
          !I8(__fort_allocated)(bb)) {
        t2 = vatd;
        is_unl_poly |= t2->tag == __POLY && t2->baseTag == __POLY;
      }
    }
    va_end(va);
    if (is_unl_poly)
      return 0;
  }

  atd = t1->type ? t1->type : (TYPE_DESC *)t1;
  btd = t2->type ? t2->type : (TYPE_DESC *)t2;
  return atd == btd ? GET_DIST_TRUE_LOG : 0;
}

/* Identical to extends_type_of() above apart from name and result type. */
__LOG8_T
ENTF90(KEXTENDS_TYPE_OF, kextends_type_of)(void *ab, OBJECT_DESC *ad, void *bb,
                                           OBJECT_DESC *bd, __INT_T flag, ...)
{
  OBJECT_DESC *t1 = ad, *t2 = bd;
  TYPE_DESC *atd, *btd;

  if (!ad || !bd)
    return 0;

  if (flag) {
    va_list va;
    int is_unl_poly_arg1 = 0, is_unl_poly_arg2 = 0;

    va_start(va, flag);
    if (flag & (ARG1_PTR | ARG1_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG1_PTR) &&
            ENTFTN(ASSOCIATED, associated)(ab, (F90_Desc *)ad, 0, 0)) &&
          !I8(__fort_allocated)(ab)) {
        t1 = vatd;
        is_unl_poly_arg1 = t1->tag == __POLY && t1->baseTag == __POLY;
      }
    }
    if (flag & (ARG2_PTR | ARG2_ALLOC)) {
      OBJECT_DESC *vatd = va_arg(va, OBJECT_DESC *);
      if (!((flag & ARG2_PTR) &&
            ENTFTN(ASSOCIATED, associated)(bb, (F90_Desc *)bd, 0, 0)) &&
          !I8(__fort_allocated)(bb)) {
        t2 = vatd;
        is_unl_poly_arg2 = t2->tag == __POLY && t2->baseTag == __POLY;
      }
    }
    va_end(va);

    if (is_unl_poly_arg2) {
      /* if second argument is unlimited polymorphic and it's
       * disassociated pointer or unallocated allocatable, then return TRUE.
       */
      return GET_DIST_TRUE_LOG;
    }
    if (is_unl_poly_arg1) {
      /* if first argument is unlimited polymorphic and it's either a
       * disassociated pointer or unallocated allocatable, then return FALSE.
       */
      return 0;
    }
  }

  atd = (t1->type /*&& t1->tag > 0 && t1->tag < __NTYPES*/) ? t1->type
                                                            : (TYPE_DESC *)t1;
  btd = (t2->type /*&& t2->tag > 0 && t2->tag < __NTYPES*/) ? t2->type
                                                            : (TYPE_DESC *)t2;
  if (atd == btd)
    return GET_DIST_TRUE_LOG;

  if (atd->obj.level > btd->obj.level) {
    TYPE_DESC *parent = get_parent_pointer(atd, btd->obj.level + 1);
    if (btd == parent)
      return GET_DIST_TRUE_LOG;
  }

  return 0;
}

void
ENTF90(SET_TYPE, set_type)(F90_Desc *dd, OBJECT_DESC *td)
{
  OBJECT_DESC *td2 = (OBJECT_DESC *)dd;
  TYPE_DESC *type = td->type;

  if (type) {
    td2->type = type;
    if (type == I8(__f03_ty_to_id)[__STR]) {
      td2->size = td->size;
    }
  } else {
    td2->type = (TYPE_DESC *)td;
  }
}

/** \brief Check whether two polymorphic types are conformable.
 *
 *  This routine is similar to the conformable routines in rdst.c, but
 *  it is for two polymorphic scalar objects instead of arrays.
 *
 *  This is needed in polymorphic allocatable assignment. If two types
 *  are conformable or the type of the left hand side expression is large
 *  enough to hold the value(s) on the right hand side, then we do not have
 *  to reallocate the left hand side if it's already allocated.
 *
 *  \param ab is the address of the first object.
 *  \param ab is the first object's descriptor.
 *  \param bd is the second object's descriptor.
 *  \param flag can be 0, 1, 2.  See flag's description for poly_asn().
 *
 *  \return 1 if types are conformable; 0 if types are not conformable but
 *          \param ab is big enough to hold \param bd; -1 if \param ab is not
 *          conformable, not big enough, or not allocated.
 */
int
ENTF90(POLY_CONFORM_TYPES, poly_conform_types)(char *ab, F90_Desc *ad,
                                               F90_Desc *bd, __INT_T flag)
{
  /* Possible return values. Do not change the integer values */
  enum {
    NOT_BIG_ENOUGH = -1, /* not conformable, not big enough */
    BIG_ENOUGH = 0,      /* not conformable but big enough */
    CONFORMABLE = 1      /* conformable */
  };

  TYPE_DESC *src_td, *dest_td;
  int src_sz, dest_sz;
  int src_is_array = 0, dest_is_array = 0;

  if (!I8(__fort_allocated)(ab)) {
    return NOT_BIG_ENOUGH;
  }

  get_source_and_dest_sizes(ad, bd, &dest_sz, &src_sz, &dest_is_array,
                            &src_is_array, &dest_td, &src_td, flag);

  if (dest_td != 0 && src_td != 0) {
    if (dest_td == src_td && dest_sz == src_sz) {
      return CONFORMABLE;
    } else if (dest_sz >= src_sz) {
      return BIG_ENOUGH;
    }
  }
  return NOT_BIG_ENOUGH;
}

void
ENTF90(TEST_AND_SET_TYPE, test_and_set_type)(F90_Desc *dd, OBJECT_DESC *td)
{
  OBJECT_DESC *td2 = (OBJECT_DESC *)dd;
  TYPE_DESC *type = td->type;

  if (type) {
    td2->type = type;
    if (type == I8(__f03_ty_to_id)[__STR]) {
      td2->size = td->size;
    }
  } else if (td->tag > 0 && td->tag < __NTYPES) {
    td2->type = (TYPE_DESC *)td;
  }
}

__INT_T
ENTF90(GET_OBJECT_SIZE, get_object_size)(F90_Desc *d)
{
  TYPE_DESC *td;
  OBJECT_DESC *od = (OBJECT_DESC *)d;

  if (!od)
    return 0;

  td = od->type;
  return td && td != I8(__f03_ty_to_id)[__STR] ? td->obj.size : od->size;
}

__INT8_T
ENTF90(KGET_OBJECT_SIZE, kget_object_size)(F90_Desc *d)
{
  TYPE_DESC *td;
  OBJECT_DESC *od = (OBJECT_DESC *)d;

  if (!od)
    return 0;

  td = od->type;
  return (__INT8_T)(td && td != I8(__f03_ty_to_id)[__STR] ? td->obj.size
                                                          : od->size);
}

/** \brief Compute address of an element in a polymorphic array.
 *
 * This routine is intended for arrays with 4 or more dimensions.
 * For 1, 2, or 3 dimensional arrays, use the specialized routines below.
 * Those routines do not incur the overhead of variable arguments.
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param variable arguments include an __INT_T* for each index of the array
 *        element that we are computing.
 */
void
ENTF90(POLY_ELEMENT_ADDR, poly_element_addr)(char *ab, F90_Desc *ad,
                                             char **result, ...)

{
  va_list va;
  __INT_T sz;
  int i, numdims;
  __INT_T index[MAXDIMS];
  __INT_T offset;
  DECL_DIM_PTRS(add);

  va_start(va, result);

  sz = ENTF90(GET_OBJECT_SIZE, get_object_size)(ad);
  numdims = F90_RANK_G(ad);

  for (i = 0; i < numdims; ++i) {
    SET_DIM_PTRS(add, ad, i);
    index[i] = *va_arg(va, __INT_T *) - F90_DPTR_LBOUND_G(add);
  }

  i = numdims - 1;
  offset = index[i];
  for (--i; i >= 0; --i) {
    SET_DIM_PTRS(add, ad, i);
    offset = index[i] + (F90_DPTR_EXTENT_G(add) * offset);
  }
  *result = ab + sz * offset;
  va_end(va);
}

/** \brief Compute address of an element in a polymorphic array (-i8 version)
 *
 * This routine is intended for arrays with 4 or more dimensions.
 * For 1, 2, or 3 dimensional arrays, use the specialized routines below.
 * Those routines do not incur the overhead of variable arguments.
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param variable arguments include an __INT_T* for each index of the array
 *        element that we are computing.
 */
void
ENTF90(KPOLY_ELEMENT_ADDR, kpoly_element_addr)(char *ab, F90_Desc *ad,
                                               char **result, ...)

{
  va_list va;
  __INT8_T sz;
  int i, numdims;
  __INT_T index[MAXDIMS];
  __INT_T offset;
  DECL_DIM_PTRS(add);

  va_start(va, result);

  sz = ENTF90(KGET_OBJECT_SIZE, kget_object_size)(ad);
  numdims = F90_RANK_G(ad);

  for (i = 0; i < numdims; ++i) {
    SET_DIM_PTRS(add, ad, i);
    index[i] = *va_arg(va, __INT_T *) - F90_DPTR_LBOUND_G(add);
  }

  i = numdims - 1;
  offset = index[i];
  for (--i; i >= 0; --i) {
    SET_DIM_PTRS(add, ad, i);
    offset = index[i] + (F90_DPTR_EXTENT_G(add) * offset);
  }
  *result = ab + sz * offset;
  va_end(va);
}

/** \brief Compute address of an element in a 1-dimensional polymorphic array.
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param ele1 is the first dimension index of the array element.
 */
void
ENTF90(POLY_ELEMENT_ADDR1, poly_element_addr1)(char *ab, F90_Desc *ad,
                                               char **result,
                                               __INT_T *ele1)
{
  __INT_T sz;
  DECL_DIM_PTRS(add);

  sz = ENTF90(GET_OBJECT_SIZE, get_object_size)(ad);
  SET_DIM_PTRS(add, ad, 0);
  *result = ab + ((*ele1 - F90_DPTR_LBOUND_G(add)) * sz);
}

/** \brief Compute address of an element in a 2-dimensional polymorphic array.
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param ele1 is the first dimension index of the array element.
 * \param ele2 is the second dimension index of the array element.
 */
void
ENTF90(POLY_ELEMENT_ADDR2, poly_element_addr2)(char *ab, F90_Desc *ad,
                                               char **result, __INT_T *ele1,
                                               __INT_T *ele2)
{
  __INT_T sz;
  __INT_T offset;
  DECL_DIM_PTRS(add);

  sz = ENTF90(GET_OBJECT_SIZE, get_object_size)(ad);
  SET_DIM_PTRS(add, ad, 1);
  offset = (*ele2 - F90_DPTR_LBOUND_G(add));
  SET_DIM_PTRS(add, ad, 0);
  offset = (*ele1 - F90_DPTR_LBOUND_G(add)) + (F90_DPTR_EXTENT_G(add) * offset);
  *result = ab + sz * offset;
}

/** \brief Compute address of an element in a 3-dimensional polymorphic array.
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param ele1 is the first dimension index of the array element.
 * \param ele2 is the second dimension index of the array element.
 * \param ele3 is the third dimension index of the array element.
 */
void
ENTF90(POLY_ELEMENT_ADDR3, poly_element_addr3)(char *ab, F90_Desc *ad,
                                               char **result, __INT_T *ele1,
                                               __INT_T *ele2, __INT_T *ele3)
{
  __INT_T sz;
  __INT_T offset;
  DECL_DIM_PTRS(add);

  sz = ENTF90(GET_OBJECT_SIZE, get_object_size)(ad);
  SET_DIM_PTRS(add, ad, 2);
  offset = (*ele3 - F90_DPTR_LBOUND_G(add));
  SET_DIM_PTRS(add, ad, 1);
  offset = (*ele2 - F90_DPTR_LBOUND_G(add)) + (F90_DPTR_EXTENT_G(add) * offset);
  SET_DIM_PTRS(add, ad, 0);
  offset = (*ele1 - F90_DPTR_LBOUND_G(add)) + (F90_DPTR_EXTENT_G(add) * offset);
  *result = ab + sz * offset;
}

/** \brief Compute address of an element in a 1-dimensional polymorphic array.
 *         (-i8 version)
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param ele1 is the first dimension index of the array element.
 */
void
ENTF90(KPOLY_ELEMENT_ADDR1, kpoly_element_addr1)(char *ab, F90_Desc *ad,
                                                 char **result, __INT_T *ele1)
{
  __INT8_T sz;
  DECL_DIM_PTRS(add);

  sz = ENTF90(KGET_OBJECT_SIZE, kget_object_size)(ad);
  SET_DIM_PTRS(add, ad, 0);
  *result = ab + ((*ele1 - F90_DPTR_LBOUND_G(add)) * sz);
}

/** \brief Compute address of an element in a 2-dimensional polymorphic array.
 *         (-i8 version)
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param ele1 is the first dimension index of the array element.
 * \param ele2 is the second dimension index of the array element.
 */
void
ENTF90(KPOLY_ELEMENT_ADDR2, kpoly_element_addr2)(char *ab, F90_Desc *ad,
                                                 char **result, __INT_T *ele1,
                                                 __INT_T *ele2)
{
  __INT8_T sz;
  __INT_T offset;
  DECL_DIM_PTRS(add);

  sz = ENTF90(KGET_OBJECT_SIZE, kget_object_size)(ad);
  SET_DIM_PTRS(add, ad, 1);
  offset = (*ele2 - F90_DPTR_LBOUND_G(add));
  SET_DIM_PTRS(add, ad, 0);
  offset = (*ele1 - F90_DPTR_LBOUND_G(add)) + (F90_DPTR_EXTENT_G(add) * offset);
  *result = ab + sz * offset;
}

/** \brief Compute address of an element in a 3-dimensional polymorphic array.
 *         (-i8 version)
 *
 * \param ab is the base address of the array.
 * \param ad is the array's descriptor.
 * \param result is the address of the pointer that will hold the result.
 * \param ele1 is the first dimension index of the array element.
 * \param ele2 is the second dimension index of the array element.
 * \param ele3 is the third dimension index of the array element.
 */
void
ENTF90(KPOLY_ELEMENT_ADDR3, kpoly_element_addr3)(char *ab, F90_Desc *ad,
                                                 char **result, __INT_T *ele1,
                                                 __INT_T *ele2, __INT_T *ele3)
{
  __INT8_T sz;
  __INT_T offset;
  DECL_DIM_PTRS(add);

  sz = ENTF90(KGET_OBJECT_SIZE, kget_object_size)(ad);
  SET_DIM_PTRS(add, ad, 2);
  offset = (*ele3 - F90_DPTR_LBOUND_G(add));
  SET_DIM_PTRS(add, ad, 1);
  offset = (*ele2 - F90_DPTR_LBOUND_G(add)) + (F90_DPTR_EXTENT_G(add) * offset);
  SET_DIM_PTRS(add, ad, 0);
  offset = (*ele1 - F90_DPTR_LBOUND_G(add)) + (F90_DPTR_EXTENT_G(add) * offset);
  *result = ab + sz * offset;
}

/** \brief Returns a type descriptor pointer of a specified ancestor of
 *         a type descriptor.
 *
 *  \param src_td is the type descriptor used to locate the ancestor type
 *                type descriptor.
 *  \param level specifies the heirarchical position in the inheritance graph
 *               of the desired ancestor type descriptor. To find its immediate
 *               parent, specify a level equal to src_td's level.
 *
 *  \return a type descriptor representing the ancestor or NULL if there is no
 *          ancestor.
 */
static TYPE_DESC *
get_parent_pointer(TYPE_DESC *src_td, __INT_T level)
{
  __INT_T offset, src_td_level;
  TYPE_DESC *parent;

  if (level <= 0 || src_td == NULL)
    return NULL;

  src_td_level = src_td->obj.level;
  if (src_td_level < 0 || level > src_td_level)
    return NULL;

  if (src_td->parents != NULL) {
    /* The parents field is filled in, so use it to get the desired parent */
    offset = (src_td_level - level) * sizeof(__POINT_T);
    parent = *((TYPE_DESC **)(((char *)src_td->parents) + offset));
  } else {
    /* The parents field is not filled in, so find the parent from the
     * src_td base pointer. The parents field is not filled in
     * when a type descriptor is created with an older compiler.
     * Note: This method does not always work if the type descriptor is
     * defined in a shared library.
     */
    offset = level * sizeof(__POINT_T);
    parent = *((TYPE_DESC **)(((char *)src_td) - offset));
  }

  return parent;
}

static void
process_final_procedures(char *area, F90_Desc *sd)
{
  /* See also Cuda Fortran version in rte/cudafor/hammer/src/dev_allo.c
   * call dev_process_final_procedures()
   */
  OBJECT_DESC *src = (OBJECT_DESC *)sd;
  TYPE_DESC *src_td, *tmp_td;
  __INT_T rank;
  FINAL_TABLE(finals);
  __LOG_T g1;
  int is_elemental;

  is_elemental = 0;
  if (src) {
    tmp_td = src->type;
    if (tmp_td && (tmp_td->obj.tag > 0 && tmp_td->obj.tag < __NTYPES)) {
      src_td = tmp_td;
    } else {
      src_td = 0;
    }
  } else {
    return;
  }

  if (!src_td)
    return;

  if (src_td->finals) {
    finals = src_td->finals;
    rank = (sd->tag == __DESC) ? F90_RANK_G(sd) : 0;

    if (rank && finals[rank]) {
      /* array case */
      (finals[rank])(area, (char *)sd);
    } else if (!rank && finals[0]) {
      /* scalar case */
      (finals[0])(area, (char *)src_td);
    } else if (finals[MAXDIMS + 1]) {
      /* elemental case */
      if (!rank) {
        (finals[MAXDIMS + 1])(area, (char *)sd);
      } else {
        is_elemental = 1;
      }
    }
  } else {
    rank = 0;
    finals = 0;
  }

  if (src_td->layout) {
    LAYOUT_DESC *ld = src_td->layout;
    F90_Desc *fd;
    char *ptr2[1] = {0};
    char *cb;
    __LOG_T g1;
    for (; ld->tag != 0; ld++) {
      if ((ld->tag != 'T' && ld->tag != 'D' && ld->tag != 'P' &&
           ld->tag != 'F') ||
          ld->offset < 0) {
        continue;
      }
      fd = (ld->desc_offset >= 0) ? (F90_Desc *)(area + ld->desc_offset) : 0;
      if (!fd && ld->tag == 'F') {
        cb = area + ld->offset;
        if (cb && !fd && ld->declType) {
          process_final_procedures(cb, (F90_Desc *)ld->declType);
        }
      } else if (fd && (fd->tag == __POLY ||
                        (fd->tag == __DESC &&
                         (fd->kind == __DERIVED || fd->kind == __POLY)))) {
        if (rank == 0) {
          __fort_bcopy((char *)ptr2, area + ld->offset, sizeof(char *));
          cb = ptr2[0];
          g1 = (fd) ? ENTFTN(ASSOCIATED, associated)(cb, fd, 0, 0) : 0;
          if ((ld->length == 0) || (!g1 && !I8(__fort_allocated)(cb))) {
            continue;
          }
          process_final_procedures(cb, fd);
        }
      }
    }
  }
  if (is_elemental && rank > 0 && finals && finals[MAXDIMS + 1]) {
    int i;
    int src_sz = sd->lsize * (size_t)src_td->obj.size;
    for (i = 0; i < src_sz; i += src_td->obj.size) {
      g1 = ENTFTN(ASSOCIATED, associated)(area + i, sd, 0, 0);
      if (!g1 && !I8(__fort_allocated)(area + i)) {
        continue;
      }
      finals[MAXDIMS + 1](area + i, (char *)src_td);
    }
  }

  if (((F90_Desc *)src_td)->tag == __POLY && src_td->obj.level > 0) {
    /* process parent finals */
    TYPE_DESC *parent = get_parent_pointer(src_td, src_td->obj.level);

    if (rank > 0) {
      int i;
      int src_sz = sd->lsize * (size_t)src_td->obj.size;
      for (i = 0; i < src_sz; i += src_td->obj.size) {
        process_final_procedures(area + i, (F90_Desc *)parent);
      }
    } else {
      process_final_procedures(area, (F90_Desc *)parent);
    }
  }
}

void
ENTF90(FINALIZE, finalize)(char *area, F90_Desc *sd)
{
  /* See also Cuda Fortran version in rte/cudafor/hammer/src/dev_allo.c
   * call DEV_FINALIZE().
   */

  process_final_procedures(area, sd);
}

void
ENTF90(DEALLOC_POLY_MBR03A, dealloc_poly_mbr03a)(F90_Desc *sd, __STAT_T *stat,
                                                 char *area, __INT_T *firsttime,
                                                 DCHAR(errmsg) DCLEN64(errmsg))
{
  OBJECT_DESC *src = (OBJECT_DESC *)sd;
  TYPE_DESC *src_td;

  if (!I8(__fort_allocated)(area))
    return;

  if (src) {
    src_td = (src->type) ? src->type : 0;
  } else {
    src_td = 0;
  }

  process_final_procedures(area, sd);

  if (src_td && src_td->layout) {
    LAYOUT_DESC *ld = src_td->layout;
    F90_Desc *fd;
    char *ptr2[1] = {0};
    char *cb;
    __LOG_T g1;

    for (; ld->tag != 0; ld++) {
      if ((ld->tag != 'T' && ld->tag != 'D' && ld->tag != 'P' &&
           ld->tag != 'F') ||
          ld->offset < 0) {
        continue;
      }
      fd = (ld->desc_offset >= 0) ? (F90_Desc *)(area + ld->desc_offset) : 0;
      if (ld->tag == 'F') {
        continue;

      } else {
        __fort_bcopy((char *)ptr2, area + ld->offset, sizeof(char *));
        cb = ptr2[0];
      }
      if (ld->tag == 'F') {
        if (ld->declType)
          process_final_procedures(cb, (F90_Desc *)ld->declType);
        continue;
      }
      g1 = (fd) ? ENTFTN(ASSOCIATED, associated)(cb, fd, 0, 0) : 0;
      if (!g1 && !I8(__fort_allocated)(cb)) {
        continue;
      }
      if (ld->tag == 'T') {
        /* Need to deallocate allocatable component */
        ENTF90(DEALLOC_MBR03, dealloc_mbr03)
        (stat, cb, firsttime, CADR(errmsg), CLEN(errmsg));
      }
    }
  }
  ENTF90(DEALLOC_MBR03, dealloc_mbr03)
  (stat, area, firsttime, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(DEALLOC_POLY_MBR03, dealloc_poly_mbr03)(F90_Desc *sd, __STAT_T *stat,
                                               char *area, __INT_T *firsttime,
                                               DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(DEALLOC_POLY_MBR03A, dealloc_poly_mbr03a)
  (sd, stat, area, firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(DEALLOC_POLY03A, dealloc_poly03a)(F90_Desc *sd, __STAT_T *stat,
                                         char *area, __INT_T *firsttime,
                                         DCHAR(errmsg) DCLEN64(errmsg))
{
  OBJECT_DESC *src = (OBJECT_DESC *)sd;
  TYPE_DESC *src_td;

  if (!I8(__fort_allocated)(area))
    return;

  if (src) {
    src_td = (src->type) ? src->type : 0;
  } else {
    src_td = 0;
  }

  process_final_procedures(area, sd);

  if (src_td && src_td->layout) {
    LAYOUT_DESC *ld = src_td->layout;
    F90_Desc *fd;
    char *ptr2[1] = {0};
    char *cb;
    __LOG_T g1;

    for (; ld->tag != 0; ld++) {
      if ((ld->tag != 'T' && ld->tag != 'D' && ld->tag != 'P' &&
           ld->tag != 'F') ||
          ld->offset < 0) {
        continue;
      }
      fd = (ld->desc_offset >= 0) ? (F90_Desc *)(area + ld->desc_offset) : 0;
      if (ld->tag == 'F') {
        continue;
      } else {
        __fort_bcopy((char *)ptr2, area + ld->offset, sizeof(char *));
        cb = ptr2[0];
      }
      g1 = (fd) ? ENTFTN(ASSOCIATED, associated)(cb, fd, 0, 0) : 0;
      if (!g1 && !I8(__fort_allocated)(cb)) {
        continue;
      }
      if (ld->tag == 'F') {
        if (ld->declType)
          process_final_procedures(cb, (F90_Desc *)ld->declType);
        continue;
      }
      if (fd) {
        if (ld->tag == 'T' && src_td->obj.tag == __POLY &&
            (fd->tag == __DESC || fd->tag == __POLY)) {
          ENTF90(DEALLOC_POLY_MBR03A, dealloc_poly_mbr03a)
          (fd, stat, cb, firsttime, CADR(errmsg), CLEN(errmsg));
        }
      }
    }
  }
  ENTF90(DEALLOC03A, dealloc03a)
  (stat, area, firsttime, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(DEALLOC_POLY03, dealloc_poly03)(F90_Desc *sd, __STAT_T *stat,
                                       char *area, __INT_T *firsttime,
                                       DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(DEALLOC_POLY03A, dealloc_poly03a)
  (sd, stat, area, firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

/* Used with F2003 sourced allocation. Allocate and assign
 * components of a derived type. This function assumes
 * that it was called by POLY_ASN() below.
 */
static void
sourced_alloc_and_assign(char *ab, char *bb, TYPE_DESC *td)
{
  LAYOUT_DESC *ld;
  char *cb, *db;
  __INT_T one = 1;
  __INT_T zero = 0;
  __INT_T len;
  __INT_T kind = __NONE;
  char *errmsg = "sourced_alloc_and_assign: malloc error";

  if (td == 0 || td->layout == 0) {
    return;
  }

  for (ld = td->layout; ld->tag != 0; ld++) {
    if ((ld->tag != 'F' && ld->tag != 'T') || ld->offset < 0) {
      continue;
    }
    if (ld->tag == 'F') {
      if (ld->declType != NULL) {
        cb = (bb + ld->offset);
        db = (ab + ld->offset);
        sourced_alloc_and_assign(db, cb, ld->declType);
      }
      continue;
    }
    cb = *(char **)(bb + ld->offset);

    if (ld->desc_offset > 0) {
      F90_Desc *fd = (F90_Desc *)(ab + ld->desc_offset);
      if (!ENTFTN(ASSOCIATED, associated)(cb, fd, 0, 0) &&
          !I8(__fort_allocated)(cb)) {
        continue;
      }
      if (fd->tag == __DESC && fd->rank > 0) {
        len = fd->lsize * fd->len;
      } else {
        len = ENTF90(GET_OBJECT_SIZE, get_object_size)(fd);
      }

      ENTF90(PTR_SRC_ALLOC03, ptr_src_alloc03)
      (fd, &one, &kind, &len, (__STAT_T *)(ENTCOMN(0, 0)), &db,
       (__POINT_T *)(ENTCOMN(0, 0)), &zero, errmsg, strlen(errmsg));
      *(char **)(ab + ld->offset) = db;
      __fort_bcopy(db, cb, len);
      if (ld->tag == 'T') {
        OBJECT_DESC *od = (OBJECT_DESC *)fd;
        if (fd->tag == __DESC && fd->rank > 0) {
          sourced_alloc_and_assign_array(fd->lsize, db, cb, od->type);
        } else {
          sourced_alloc_and_assign(db, cb, od->type);
        }
      }
    } else if ((len = ld->length) > 0) {
      ENTF90(PTR_ALLOC03, ptr_alloc03)
      (&one, &kind, &len, (__STAT_T *)(ENTCOMN(0, 0)), &db,
       (__POINT_T *)(ENTCOMN(0, 0)), &zero, errmsg, strlen(errmsg));
      *(char **)(ab + ld->offset) = db;
      if (I8(__fort_allocated)(cb)) {
        __fort_bcopy(db, cb, len);
        if (ld->tag == 'T') {
          sourced_alloc_and_assign(db, cb, ld->declType);
        }
      }
    }
  }
}

/** \brief Perform sourced allocation and assign on each element of an array.
 *
 *  \param extent is the number of elements to allocate and assign.
 *  \param ab is a pointer to the destination array.
 *  \param bb is a pointer to the source array.
 *  \param td is the destination array's type descriptor.
 */
static void
sourced_alloc_and_assign_array(int extent, char *ab, char *bb, TYPE_DESC *td)
{
  if (td != 0) {
    const int elem_size = td->obj.size;
    const int end_offset = extent * elem_size;
    int elem_offset;
    for (elem_offset = 0; elem_offset < end_offset; elem_offset += elem_size) {
      sourced_alloc_and_assign(ab + elem_offset, bb + elem_offset, td);
    }
  }
}

/** \brief Same as sourced_alloc_and_assign_array() except the source
 *         argument is a scalar; not an array.
 *
 *  \param extent is the number of elements to allocate and assign.
 *  \param ab is a pointer to the destination array.
 *  \param bb is a pointer to the source array.
 *  \param td is the destination array's type descriptor.
 */
static void
sourced_alloc_and_assign_array_from_scalar(int extent, char *ab, char *bb,
                                           TYPE_DESC *td)
{
  if (td != 0) {
    const int elem_size = td->obj.size;
    const int end_offset = extent * elem_size;
    int elem_offset;
    for (elem_offset = 0; elem_offset < end_offset; elem_offset += elem_size) {
      sourced_alloc_and_assign(ab + elem_offset, bb, td);
    }
  }
}

/** \brief Computes destination/first object and source/second object  sizes
 *         and other variables used by the poly_asn() and poly_conform_types()
 *         routines.
 *
 *  \param ad is the destination/first descriptor in a polymorphic assignment
 *         or polymorphic type conformance test.
 *  \param bd is the source/second descriptor in a polymorphic assignment or
 *         polymorphic type conformance test.
 *  \param dest_sz is used to return the destination/first object's size.
 *  \param src_sz is used to return the source/second object's size.
 *  \param dest_is_array stores whether the destination/first object is an
 *         array.
 *  \param src_is_array stores whether the source/second object is an array.
 *  \param tad is used to return the destination/first object's type descriptor.
 *  \param tbd is used to return the source/second object's type descriptor.
 *  \param flag can be 0, 1, 2.  See flag's description for poly_asn().
 */
static void
get_source_and_dest_sizes(F90_Desc *ad, F90_Desc *bd, int *dest_sz, int *src_sz,
                          int *dest_is_array, int *src_is_array,
                          TYPE_DESC **tad, TYPE_DESC **tbd, __INT_T flag)
{
  OBJECT_DESC *src = (OBJECT_DESC *)bd;
  OBJECT_DESC *dest = (OBJECT_DESC *)ad;
  TYPE_DESC *src_td, *dest_td;

  *dest_is_array = *src_is_array = 0;

  if (dest) {
    dest_td = dest->type ? dest->type : (TYPE_DESC *)ad;
  } else {
    dest_td = 0;
  }

  if (src && (flag || src->tag == __DESC || src->tag == __POLY)) {
    src_td = src->type ? src->type : (TYPE_DESC *)bd;
  } else {
    src_td = 0;
  }

  if (src_td) {
    if (bd && bd->tag == __DESC && bd->rank > 0) {
      *src_sz = bd->lsize * (size_t)src_td->obj.size;
      *src_is_array = 1;
    } else if (src_td->obj.baseTag == __STR) {
      *src_sz = (size_t)(ad->len * ad->lsize);
      *src_is_array = 1;
    } else if (bd && (flag || bd->tag == __POLY || bd->tag == __DESC)) {
      *src_sz = (size_t)src_td->obj.size;
    } else {
      *src_sz = 0;
    }
  } else if (bd && !flag && ISSCALAR(bd) && bd->tag != __POLY &&
             bd->tag != __STR && bd->tag < __NTYPES) {
#if defined(_WIN64)
    *src_sz = __get_fort_size_of(bd->tag);
#else
    *src_sz = __fort_size_of[bd->tag];
#endif
  } else {
    *src_sz = 0;
  }

  if (dest_td) {
    if (ad && ad->tag == __DESC && ad->rank > 0) {
      *dest_sz = ad->lsize * (size_t)dest_td->obj.size;
      *dest_is_array = 1;
    } else if (ad && ad->tag == __DESC && dest_td &&
               dest_td->obj.tag == __POLY && ad->len > 0 && !ad->lsize &&
               !ad->gsize && ad->kind > 0 && ad->kind < __NTYPES) {
      *dest_sz = (size_t)dest_td->obj.size * ad->len;
    } else if (!*src_sz || ((flag == 1 || (ad && ad->tag == __DESC)) &&
                            dest_td->obj.tag == __POLY)) {
      *dest_sz = dest_td != I8(__f03_ty_to_id)[__STR]
                     ? (size_t)dest_td->obj.size
                     : ad->len;
    } else {
      *dest_sz = 0;
    }
  } else {
    *dest_sz = 0;
  }

  *tad = dest_td;
  *tbd = src_td;
}

void
ENTF90(POLY_ASN, poly_asn)(char *ab, F90_Desc *ad, char *bb, F90_Desc *bd,
                           __INT_T flag)
{
  /* Copy the contents of object bb to object ab
   * Assumes destination descriptor, ad, is a full descriptor.
   * If flag == 0, then source descriptor, bd, is a scalar "fake" descriptor
   * If flag == 1, assume full descriptor for bd
   * If flag == 2, assume full descriptor for bd and copy bd into ad.
   */

  OBJECT_DESC *src = (OBJECT_DESC *)bd;
  TYPE_DESC *src_td, *dest_td;
  int src_sz, dest_sz, sz;
  int dest_is_array, src_is_array, i;

  get_source_and_dest_sizes(ad, bd, &dest_sz, &src_sz, &dest_is_array,
                            &src_is_array, &dest_td, &src_td, flag);

  if (src_sz && src_td && src_td->obj.tag == __POLY &&
      (!ad || ad->tag != __DESC || !dest_td || dest_td->obj.tag != __POLY))
    sz = src_sz;
  else if (!src_sz ||
           (ad && ad->tag == __DESC && dest_td && dest_td->obj.tag == __POLY &&
            (!src_td || src_td->obj.tag != __POLY)))
    sz = dest_sz;
  else
    sz = (src_sz > dest_sz) ? src_sz : dest_sz;

  if (src_td && src_td->obj.size && dest_is_array && !src_is_array) {
    for (i = 0; i < dest_sz; i += src_td->obj.size) {
      __fort_bcopy(ab + i, bb, src_sz);
    }
  } else if (src_sz && !flag && ISSCALAR(bd) && bd->tag != __POLY &&
             bd->tag < __NTYPES) {
    for (i = 0; i < sz; i += src_sz) {
      __fort_bcopy(ab + i, bb, src_sz);
    }
  } else {
    __fort_bcopy(ab, bb, sz);
  }

  if (flag && ad && bd && ad != bd && F90_TAG_G(bd) == __DESC &&
      (F90_TAG_G(ad) == __DESC || flag == 2)) {
    __fort_bcopy((char *)ad, (char *)bd,
                 SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(bd)));
    SET_F90_DIST_DESC_PTR(ad, F90_RANK_G(ad));
    /* check for align-target to self */
    if (DIST_ALIGN_TARGET_G(ad) == ad) {
      DIST_ALIGN_TARGET_P(ad, ad);
    }
  } else if (flag > 0 && src_td) {
    ENTF90(SET_TYPE, set_type)(ad, src);
    dest_td = src_td;
  }

  if (flag) {
    if (src_td && (src_td->obj.tag > 0 && src_td->obj.tag < __NTYPES) &&
        !src_is_array && !dest_is_array) {
      sourced_alloc_and_assign(ab, bb, src_td->obj.type);
    } else if (dest_is_array && src_is_array) {
      sourced_alloc_and_assign_array(ad->lsize, ab, bb, dest_td->obj.type);
    } else if (dest_is_array) {
      sourced_alloc_and_assign_array_from_scalar(ad->lsize, ab, bb,
                                                 dest_td->obj.type);
    }
  }
}

void
I8(__fort_dump_type)(TYPE_DESC *d)
{
  fprintf(__io_stderr(), "Polymorphic variable type '");
  switch (d->obj.baseTag) {
  case __NONE:
    fprintf(__io_stderr(), "__NONE'\n");
    return;
  case __SHORT:
    fprintf(__io_stderr(), "__SHORT'\n");
    break;
  case __USHORT:
    fprintf(__io_stderr(), "__USHORT'\n");
    break;
  case __CINT:
    fprintf(__io_stderr(), "__CINT'\n");
    break;
  case __UINT:
    fprintf(__io_stderr(), "__UINT'\n");
    break;
  case __LONG:
    fprintf(__io_stderr(), "__LONG'\n");
    break;
  case __ULONG:
    fprintf(__io_stderr(), "__FLOAT'\n");
    break;
  case __DOUBLE:
    fprintf(__io_stderr(), "__DOUBLE'\n");
    break;
  case __CPLX8:
    fprintf(__io_stderr(), "__CPLX8'\n");
    break;
  case __CPLX16:
    fprintf(__io_stderr(), "__CPLX16'\n");
    break;
  case __CHAR:
    fprintf(__io_stderr(), "__CHAR'\n");
    break;
  case __UCHAR:
    fprintf(__io_stderr(), "__UCHAR'\n");
    break;
  case __LONGDOUBLE:
    fprintf(__io_stderr(), "__LONGDOUBLE'\n");
    break;
  case __STR:
    fprintf(__io_stderr(), "__STR'\n");
    break;
  case __LONGLONG:
    fprintf(__io_stderr(), "__LONGLONG'\n");
    break;
  case __ULONGLONG:
    fprintf(__io_stderr(), "__ULONGLONG'\n");
    break;
  case __LOG1:
    fprintf(__io_stderr(), "__LOG1'\n");
    break;
  case __LOG2:
    fprintf(__io_stderr(), "__LOG2'\n");
    break;
  case __LOG4:
    fprintf(__io_stderr(), "__LOG4'\n");
    break;
  case __LOG8:
    fprintf(__io_stderr(), "__LOG8'\n");
    break;
  case __WORD4:
    fprintf(__io_stderr(), "__WORD4'\n");
    break;
  case __WORD8:
    fprintf(__io_stderr(), "__WORD8'\n");
    break;
  case __NCHAR:
    fprintf(__io_stderr(), "__NCHAR'\n");
    break;
  case __INT2:
    fprintf(__io_stderr(), "__INT2'\n");
    break;
  case __INT4:
    fprintf(__io_stderr(), "__INT4'\n");
    break;
  case __INT8:
    fprintf(__io_stderr(), "__INT8'\n");
    break;
  case __REAL4:
    fprintf(__io_stderr(), "__REAL4'\n");
    break;
  case __REAL8:
    fprintf(__io_stderr(), "__REAL8'\n");
    break;
  case __REAL16:
    fprintf(__io_stderr(), "__REAL16'\n");
    break;
  case __CPLX32:
    fprintf(__io_stderr(), "__CPLX32'\n");
    break;
  case __WORD16:
    fprintf(__io_stderr(), "__WORD16'\n");
    break;
  case __INT1:
    fprintf(__io_stderr(), "__INT1'\n");
    break;
  case __DERIVED:
    fprintf(__io_stderr(), "__DERIVED'\n");
    break;
  case __PROC:
    fprintf(__io_stderr(), "__PROC'\n");
    break;
  case __DESC:
    fprintf(__io_stderr(), "__DESC'\n");
    break;
  case __SKED:
    fprintf(__io_stderr(), "__SKED'\n");
    break;
  case __M128:
    fprintf(__io_stderr(), "__M128'\n");
    break;
  case __M256:
    fprintf(__io_stderr(), "__M256'\n");
    break;
  case __INT16:
    fprintf(__io_stderr(), "__INT16'\n");
    break;
  case __LOG16:
    fprintf(__io_stderr(), "__LOG16'\n");
    break;
  case __QREAL16:
    fprintf(__io_stderr(), "__QREAL16'\n");
    break;
  case __QCPLX32:
    fprintf(__io_stderr(), "__QCPLX32'\n");
    break;
  case __POLY:
    fprintf(__io_stderr(), "__POLY'\n");
    break;
  case __PROCPTR:
    fprintf(__io_stderr(), "__PROCPTR'\n");
    break;
  default:
    fprintf(__io_stderr(), "unknown (%d)'\n", d->obj.baseTag);
    return;
  }

  fprintf(__io_stderr(), "Size: %d\n", d->obj.size);
  fprintf(__io_stderr(), "Type Descriptor:\n\t'%s'\n", d->name);
  if (d->obj.level > 0) {
    __INT_T level;
    fprintf(__io_stderr(), "(Child Type)\n");
    fprintf(__io_stderr(), "Parent Descriptor%s\n",
            (d->obj.level == 1) ? ":" : "s:");
    for (level = d->obj.level - 1; level >= 0; --level) {
      TYPE_DESC *parent = get_parent_pointer(d, level + 1);
      fprintf(__io_stderr(), "\t'%s'\n", parent->name);
    }

    if (d->func_table) {
      fprintf(__io_stderr(), "function table: %p\n", *(d->func_table));
    }

  } else
    fprintf(__io_stderr(), "(Base Type)\n");

  if (d->layout != 0) {
    LAYOUT_DESC *ld;
    fprintf(__io_stderr(), "Layout descriptors:\n");
    for (ld = d->layout; ld->tag != 0; ld++) {
      if ((/*ld->tag != 'P' &&*/ ld->tag != 'T') || ld->offset < 0) {
        continue;
      }
      fprintf(__io_stderr(),
              "  tag=%c offset=%d desc_offset=%d length=%d declType=%p\n",
              ld->tag, ld->offset, ld->desc_offset, ld->length, ld->declType);
    }
  }
}

static struct type_desc I8(__f03_short_td) = {
    {__POLY, __SHORT, 0, sizeof(__SHORT_T), 0, 0, 0, 0, 0, &I8(__f03_short_td)},
    0,
    0,
    0,
    0,
    "__f03_short_td"};

static struct type_desc I8(__f03_ushort_td) = {{__POLY, __USHORT, 0,
                                                sizeof(__USHORT_T), 0, 0, 0, 0,
                                                0, &I8(__f03_ushort_td)},
                                               0,
                                               0,
                                               0,
                                               0,
                                               "__f03_ushort_td"};

static struct type_desc I8(__f03_cint_td) = {
    {__POLY, __CINT, 0, sizeof(__CINT_T), 0, 0, 0, 0, 0, &I8(__f03_cint_td)},
    0,
    0,
    0,
    0,
    "__f03_cint_td"};

static struct type_desc I8(__f03_uint_td) = {
    {__POLY, __UINT, 0, sizeof(__UINT_T), 0, 0, 0, 0, 0, &I8(__f03_uint_td)},
    0,
    0,
    0,
    0,
    "__f03_uint_td"};

static struct type_desc I8(__f03_long_td) = {
    {__POLY, __LONG, 0, sizeof(__LONG_T), 0, 0, 0, 0, 0, &I8(__f03_long_td)},
    0,
    0,
    0,
    0,
    "__f03_long_td"};

static struct type_desc I8(__f03_ulong_td) = {
    {__POLY, __ULONG, 0, sizeof(__ULONG_T), 0, 0, 0, 0, 0, &I8(__f03_ulong_td)},
    0,
    0,
    0,
    0,
    "__f03_ulong_td"};

static struct type_desc I8(__f03_float_td) = {
    {__POLY, __FLOAT, 0, sizeof(__FLOAT_T), 0, 0, 0, 0, 0, &I8(__f03_float_td)},
    0,
    0,
    0,
    0,
    "__f03_float_td"};

static struct type_desc I8(__f03_double_td) = {{__POLY, __DOUBLE, 0,
                                                sizeof(__DOUBLE_T), 0, 0, 0, 0,
                                                0, &I8(__f03_double_td)},
                                               0,
                                               0,
                                               0,
                                               0,
                                               "__f03_double_td"};

static struct type_desc I8(__f03_cplx8_td) = {
    {__POLY, __CPLX8, 0, sizeof(__CPLX8_T), 0, 0, 0, 0, 0, &I8(__f03_cplx8_td)},
    0,
    0,
    0,
    0,
    "__f03_cplx8_td"};

static struct type_desc I8(__f03_cplx16_td) = {{__POLY, __CPLX16, 0,
                                                sizeof(__CPLX16_T), 0, 0, 0, 0,
                                                0, &I8(__f03_cplx16_td)},
                                               0,
                                               0,
                                               0,
                                               0,
                                               "__f03_cplx16_td"};

static struct type_desc I8(__f03_char_td) = {
    {__POLY, __CHAR, 0, sizeof(__CHAR_T), 0, 0, 0, 0, 0, &I8(__f03_char_td)},
    0,
    0,
    0,
    0,
    "__f03_char_td"};

static struct type_desc I8(__f03_uchar_td) = {
    {__POLY, __UCHAR, 0, sizeof(__UCHAR_T), 0, 0, 0, 0, 0, &I8(__f03_uchar_td)},
    0,
    0,
    0,
    0,
    "__f03_uchar_td"};

static struct type_desc I8(__f03_longdouble_td) = {
    {__POLY, __LONGDOUBLE, 0, sizeof(__LONGDOUBLE_T), 0, 0, 0, 0, 0,
     &I8(__f03_longdouble_td)},
    0,
    0,
    0,
    0,
    "__f03_longdouble_td"};

static struct type_desc I8(__f03_str_td) = {
    {__POLY, __STR, 0, sizeof(__STR_T), 0, 0, 0, 0, 0, &I8(__f03_str_td)},
    0,
    0,
    0,
    0,
    "__f03_str_td"};

static struct type_desc I8(__f03_longlong_td) = {{__POLY, __LONGLONG, 0,
                                                  sizeof(__LONGLONG_T), 0, 0, 0,
                                                  0, 0, &I8(__f03_longlong_td)},
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 "__f03_longlong_td"};

static struct type_desc I8(__f03_ulonglong_td) = {
    {__POLY, __ULONGLONG, 0, sizeof(__ULONGLONG_T), 0, 0, 0, 0, 0,
     &I8(__f03_ulonglong_td)},
    0,
    0,
    0,
    0,
    "__f03_ulonglong_td"};

static struct type_desc I8(__f03_log1_td) = {
    {__POLY, __LOG1, 0, sizeof(__LOG1_T), 0, 0, 0, 0, 0, &I8(__f03_log1_td)},
    0,
    0,
    0,
    0,
    "__f03_log1_td"};

static struct type_desc I8(__f03_log2_td) = {
    {__POLY, __LOG2, 0, sizeof(__LOG2_T), 0, 0, 0, 0, 0, &I8(__f03_log2_td)},
    0,
    0,
    0,
    0,
    "__f03_log2_td"};

static struct type_desc I8(__f03_log4_td) = {
    {__POLY, __LOG4, 0, sizeof(__LOG4_T), 0, 0, 0, 0, 0, &I8(__f03_log4_td)},
    0,
    0,
    0,
    0,
    "__f03_log4_td"};

static struct type_desc I8(__f03_log8_td) = {
    {__POLY, __LOG8, 0, sizeof(__LOG8_T), 0, 0, 0, 0, 0, &I8(__f03_log8_td)},
    0,
    0,
    0,
    0,
    "__f03_log8_td"};

static struct type_desc I8(__f03_word4_td) = {
    {__POLY, __WORD4, 0, sizeof(__WORD4_T), 0, 0, 0, 0, 0, &I8(__f03_word4_td)},
    0,
    0,
    0,
    0,
    "__f03_word4_td"};

static struct type_desc I8(__f03_word8_td) = {
    {__POLY, __WORD8, 0, sizeof(__WORD8_T), 0, 0, 0, 0, 0, &I8(__f03_word8_td)},
    0,
    0,
    0,
    0,
    "__f03_word8_td"};

static struct type_desc I8(__f03_nchar_td) = {
    {__POLY, __NCHAR, 0, sizeof(__NCHAR_T), 0, 0, 0, 0, 0, &I8(__f03_nchar_td)},
    0,
    0,
    0,
    0,
    "__f03_nchar_td"};

static struct type_desc I8(__f03_int2_td) = {
    {__POLY, __INT2, 0, sizeof(__INT2_T), 0, 0, 0, 0, 0, &I8(__f03_int2_td)},
    0,
    0,
    0,
    0,
    "__f03_int2_td"};

static struct type_desc I8(__f03_int4_td) = {
    {__POLY, __INT4, 0, sizeof(__INT4_T), 0, 0, 0, 0, 0, &I8(__f03_int4_td)},
    0,
    0,
    0,
    0,
    "__f03_int4_td"};

static struct type_desc I8(__f03_int8_td) = {
    {__POLY, __INT8, 0, sizeof(__INT8_T), 0, 0, 0, 0, 0, &I8(__f03_int8_td)},
    0,
    0,
    0,
    0,
    "__f03_int8_td"};

static struct type_desc I8(__f03_real4_td) = {
    {__POLY, __REAL4, 0, sizeof(__REAL4_T), 0, 0, 0, 0, 0, &I8(__f03_real4_td)},
    0,
    0,
    0,
    0,
    "__f03_real4_td"};

static struct type_desc I8(__f03_real8_td) = {
    {__POLY, __REAL8, 0, sizeof(__REAL8_T), 0, 0, 0, 0, 0, &I8(__f03_real8_td)},
    0,
    0,
    0,
    0,
    "__f03_real8_td"};

static struct type_desc I8(__f03_real16_td) = {{__POLY, __REAL16, 0,
                                                sizeof(__REAL16_T), 0, 0, 0, 0,
                                                0, &I8(__f03_real16_td)},
                                               0,
                                               0,
                                               0,
                                               0,
                                               "__f03_real16_td"};

static struct type_desc I8(__f03_cplx32_td) = {{__POLY, __CPLX32, 0,
                                                sizeof(__CPLX32_T), 0, 0, 0, 0,
                                                0, &I8(__f03_cplx32_td)},
                                               0,
                                               0,
                                               0,
                                               0,
                                               "__f03_cplx32_td"};

static struct type_desc I8(__f03_word16_td) = {{__POLY, __WORD16, 0,
                                                sizeof(__WORD16_T), 0, 0, 0, 0,
                                                0, &I8(__f03_word16_td)},
                                               0,
                                               0,
                                               0,
                                               0,
                                               "__f03_word16_td"};

static struct type_desc I8(__f03_int1_td) = {
    {__POLY, __INT1, 0, sizeof(__INT1_T), 0, 0, 0, 0, 0, &I8(__f03_int1_td)},
    0,
    0,
    0,
    0,
    "__f03_int1_td"};

/* The order of the type descriptors below must correspond with their type
 * code in _DIST_TYPE enum defined in pghpft.h
 */
static struct type_desc *I8(__f03_ty_to_id)[__NTYPES] = {
    0,
    &I8(__f03_short_td),
    &I8(__f03_ushort_td),
    &I8(__f03_cint_td),
    &I8(__f03_uint_td),
    &I8(__f03_long_td),
    &I8(__f03_ulong_td),
    &I8(__f03_float_td),
    &I8(__f03_double_td),
    &I8(__f03_cplx8_td),
    &I8(__f03_cplx16_td),
    &I8(__f03_char_td),
    &I8(__f03_uchar_td),
    &I8(__f03_longdouble_td),
    &I8(__f03_str_td),
    &I8(__f03_longlong_td),
    &I8(__f03_ulonglong_td),
    &I8(__f03_log1_td),
    &I8(__f03_log2_td),
    &I8(__f03_log4_td),
    &I8(__f03_log8_td),
    &I8(__f03_word4_td),
    &I8(__f03_word8_td),
    &I8(__f03_nchar_td),
    &I8(__f03_int2_td),
    &I8(__f03_int4_td),
    &I8(__f03_int8_td),
    &I8(__f03_real4_td),
    &I8(__f03_real8_td),
    &I8(__f03_real16_td),
    &I8(__f03_cplx32_td),
    &I8(__f03_word16_td),
    &I8(__f03_int1_td),
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0};

void
ENTF90(SET_INTRIN_TYPE, set_intrin_type)(F90_Desc *dd, __INT_T intrin_type)
{
  OBJECT_DESC *od = (OBJECT_DESC *)dd;

#if DEBUG
  if (!od)
    return;

  od->type = (intrin_type >= 0 && intrin_type < __NTYPES)
                 ? I8(__f03_ty_to_id)[intrin_type]
                 : 0;

  if (od->type == 0) {
    __fort_abort("set_intrin_type: Illegal intrinsic type");
  }
#else
  od->type = I8(__f03_ty_to_id)[intrin_type];
#endif
}

__LOG_T
ENTF90(SAME_INTRIN_TYPE_AS, same_intrin_type_as)(void *ab, OBJECT_DESC *ad,
                                                 void *bb, __INT_T intrin_type,
                                                 __INT_T flag, ...)
{
  TYPE_DESC *btd;
  OBJECT_DESC *t1, *t2;
  __LOG_T g1;
  va_list va;

  if (!ad)
    return 0;

  if (flag) {
    va_start(va, flag);
    if (flag & ARG1_PTR) {
      /* first arg is a pointer */

      g1 = ENTFTN(ASSOCIATED, associated)(ab, (F90_Desc *)ad, 0, 0);
      if (!g1 && !I8(__fort_allocated)(ab)) {
        t1 = va_arg(va, OBJECT_DESC *); /* get declared type */
      } else {
        t1 = ad; /* use dynamic (runtime) type */
      }
    } else if (flag & ARG1_ALLOC) {
      /* first arg is an allocatable */
      g1 = I8(__fort_allocated)(ab);
      if (!g1) {
        t1 = va_arg(va, OBJECT_DESC *); /* get declared type */
      } else {
        t1 = ad; /* use dynamic (runtime) type */
      }
    } else {
      t1 = ad; /* use dynamic (runtime) type */
    }
  } else {
    t1 = ad;
  }

  btd = I8(__f03_ty_to_id)[intrin_type];
  t2 = &(btd->obj);

  return ENTF90(SAME_TYPE_AS, same_type_as)(ab, t1, bb, t2, 0);
}

__LOG8_T
ENTF90(KSAME_INTRIN_TYPE_AS, ksame_intrin_type_as)(void *ab, OBJECT_DESC *ad,
                                                   void *bb,
                                                   __INT_T intrin_type,
                                                   __INT_T flag, ...)
{
  TYPE_DESC *btd;
  OBJECT_DESC *t1, *t2;
  __LOG_T g1;
  va_list va;

  if (!ad)
    return 0;

  if (flag) {
    va_start(va, flag);
    if (flag & ARG1_PTR) {
      /* first arg is a pointer */

      g1 = ENTFTN(ASSOCIATED, associated)(ab, (F90_Desc *)ad, 0, 0);
      if (!g1 && !I8(__fort_allocated)(ab)) {
        t1 = va_arg(va, OBJECT_DESC *); /* get declared type */
      } else {
        t1 = ad; /* use dynamic (runtime) type */
      }
    } else if (flag & ARG1_ALLOC) {
      /* first arg is an allocatable */
      g1 = I8(__fort_allocated)(ab);
      if (!g1) {
        t1 = va_arg(va, OBJECT_DESC *); /* get declared type */
      } else {
        t1 = ad; /* use dynamic (runtime) type */
      }
    } else {
      t1 = ad; /* use dynamic (runtime) type */
    }

  } else {
    t1 = ad;
  }

  btd = I8(__f03_ty_to_id)[intrin_type];
  t2 = &(btd->obj);

  return ENTF90(KSAME_TYPE_AS, ksame_type_as)(ab, t1, bb, t2, 0);
}

void
ENTF90(POLY_ASN_SRC_INTRIN, poly_asn_src_intrin)(char *ab, F90_Desc *ad,
                                                 char *bb, __INT_T intrin_type,
                                                 __INT_T flag)
{
  F90_Desc *bd;
  TYPE_DESC *td;

  td = I8(__f03_ty_to_id)[intrin_type];
#if DEBUG
  if (td == 0) {
    __fort_abort("poly_asn_src_intrin: Illegal intrinsic type");
  }
#endif
  bd = (F90_Desc *)&(td->obj);

  ENTF90(POLY_ASN, poly_asn)(ab, ad, bb, bd, flag);
}

void
ENTF90(POLY_ASN_DEST_INTRIN, poly_asn_dest_intrin)(char *ab,
                                                   __INT_T intrin_type,
                                                   char *bb, F90_Desc *bd,
                                                   __INT_T flag)
{
  F90_Desc *ad;
  TYPE_DESC *td;

  td = I8(__f03_ty_to_id)[intrin_type];
#if DEBUG
  if (td == 0) {
    __fort_abort("poly_asn_dest_intrin: Illegal intrinsic type");
  }
#endif
  ad = (F90_Desc *)&(td->obj);

  ENTF90(POLY_ASN, poly_asn)(ab, ad, bb, bd, flag);
}

/** \brief This routine checks whether a descriptor is associated with an
 *         intrinsic type.
 *
 *  \param dd is the descriptor we are testing.
 *
 *  \return 1 if \param dd is associated with an intinsinc type, else 0.
 */
static int
has_intrin_type(F90_Desc *dd)
{
  int i;
  OBJECT_DESC *td = (OBJECT_DESC *)dd;

  if (td->type == NULL)
    return 0;

  for (i = 0; i < __NTYPES; ++i) {
    if (td->type == I8(__f03_ty_to_id)[i]) {
      return 1;
    }
  }

  return 0;
}

void
ENTF90(INIT_UNL_POLY_DESC, init_unl_poly_desc)(F90_Desc *dd, F90_Desc *sd,
                                               __INT_T kind)
{
  if (sd && F90_TAG_G(sd) == __DESC) {
    __fort_bcopy((char *)dd, (char *)sd,
                 SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(sd)));
    SET_F90_DIST_DESC_PTR(dd, F90_RANK_G(dd));
    /* check for align-target to self */
    if (DIST_ALIGN_TARGET_G(dd) == dd) {
      DIST_ALIGN_TARGET_P(dd, dd);
    }
    dd->kind = kind;
  } else {
    dd->len = (sd && (sd->tag == __DESC || sd->tag == __POLY)) ? sd->len : 0;
    dd->tag = __POLY;
    dd->rank = 0;
    dd->lsize = 0;
    dd->gsize = 0;
    dd->kind = kind;
    if (sd && (sd->tag == __DESC || sd->tag == __POLY || has_intrin_type(sd))) {
      ENTF90(SET_TYPE, set_type)(dd, sd);
    }
  }
}

void
ENTF90(INIT_FROM_DESC, init_from_desc)(void *object, const F90_Desc *desc,
                                       int rank)
{
  if (object && desc) {
    const OBJECT_DESC *obj_desc = (const OBJECT_DESC *)desc;
    size_t items = 1;
    size_t index[MAXDIMS];
    const TYPE_DESC *type_desc = obj_desc->type;
    int j;
    size_t element_bytes = 0;
    void *prototype = NULL;

    if (desc->tag == __DESC) {
      if (desc->rank < rank)
        rank = desc->rank;
      if (rank > 0) {
        items = desc->lsize;
        for (j = 0; j < rank; ++j) {
          index[j] = 0;
        }
      }
    }

    if (type_desc)
      obj_desc = &type_desc->obj;
    else
      type_desc = (const TYPE_DESC *)obj_desc;
    element_bytes = obj_desc->size;
    prototype = obj_desc->prototype;

    while (items-- > 0) {
      int do_increment = 1;
      char *element = object;
      size_t offset = 0;
      for (j = 0; j < rank; ++j) {
        offset += index[j] * desc->dim[j].lstride;
        if (do_increment) {
          if (++index[j] >= (size_t)desc->dim[j].extent)
            index[j] = 0;
          else
            do_increment = 0;
        }
      }
      element = (char *)object + element_bytes * offset;
      if (prototype)
        memcpy(element, prototype, element_bytes);
      else
        memset(element, 0, element_bytes);
    }
  }
}

void
ENTF90(ASN_CLOSURE, asn_closure)(PROC_DESC *pdesc, void *closure)
{
  pdesc->tag = __PROCPTR;
  pdesc->closure = closure;
}

void
ENTF90(COPY_PROC_DESC, copy_proc_desc)(PROC_DESC *dd, PROC_DESC *sd)
{
  dd->tag = __PROCPTR;
  dd->closure = sd->closure;
}
