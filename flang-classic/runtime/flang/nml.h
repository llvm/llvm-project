/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Extern decls, definitions and structs common to nmlread.c and
 * nmlwrite.c
 */

#define GBL_SIZE 5
#define MAX_DIM 30

#define ACTUAL_NDIMS(adims)                                                    \
  adims = (descp->ndims >= MAX_DIM) ? (descp->ndims - MAX_DIM) : (descp->ndims)


typedef struct {
  __POINT_T nlen;
  POINT(char, group);
  __POINT_T ndesc;
  /* 1 or more records of type NML_ITEM ... */
} NML_GROUP;

typedef struct {
  __POINT_T nlen;
  POINT(char, sym);
  POINT(char, addr);
  __POINT_T type;
  __POINT_T len;
  __POINT_T ndims;
  /*  0 or more words of dimension info appear here  */
  /*  also defined io if any  */
} NML_DESC;


#ifdef DESC_I8
typedef struct {
  __INT_T lwb;
  __INT_T upb;
  __INT_T stride;
} TRIPLE;

/* structure to assist computing the subscripts of an array section  */
typedef struct {
  int v;           /* array section's VRF index */
  int ndims;       /* number of dimension in the array */
  __INT_T elemsz;  /* size of each element */
  __INT_T idx[7];  /* current index values for each dimension */
  TRIPLE sect[7];  /* lower : upper : stride for each dimensin */
  __INT_T mult[7]; /* multiplier for each section */
  __INT_T lwb[7];  /* declared lower bound for each dimension */
  char *loc_addr;  /* array's base address */
} SB;

#else

typedef struct {
  __BIGINT_T lwb;
  __BIGINT_T upb;
  __BIGINT_T stride;
} TRIPLE;

/* structure to assist computing the subscripts of an array section  */
typedef struct {
  int v;              /* array section's VRF index */
  int ndims;          /* number of dimension in the array */
  __BIGINT_T elemsz;  /* size of each element */
  __BIGINT_T idx[7];  /* current index values for each dimension */
  TRIPLE sect[7];     /* lower : upper : stride for each dimensin */
  __BIGINT_T mult[7]; /* multiplier for each section */
  __BIGINT_T lwb[7];  /* declared lower bound for each dimension */
  char *loc_addr;     /* array's base address */
} SB;

#endif

typedef struct {
  int size;
  int avl;
  TRIPLE *base;
} TRI;


/** \brief
 * Return the size of a name list item
 */
__BIGINT_T I8(siz_of)(NML_DESC *descp);

/** \brief
 * Return the number of elements in a name list item.
 */
int nelems_of(NML_DESC *descp);

/** \brief
 * If a descriptor exists for a data item return a pointer to it.  Otherwise
 * return NULL.
 */
F90_Desc *get_descriptor(NML_DESC *descp);

