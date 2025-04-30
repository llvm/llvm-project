/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef DTYPEUTL_H_
#define DTYPEUTL_H_

#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include <stdio.h>

/**
   \brief ...
 */
ISZ_T ad_val_of(int sym);

/**
   \brief ...
 */
ISZ_T extent_of(DTYPE dtype);

/// \brief Given a constant symbol, return its numerical value.
ISZ_T get_bnd_cval(int con);

/**
   \brief ...
 */
ISZ_T size_of(DTYPE dtype);

/**
   \brief ...
 */
ISZ_T size_of_sym(SPTR sym);

/**
   \brief ...
 */
ISZ_T zsize_of(DTYPE dtype);

/**
 * \brief Return true if the data types for two functions are compatible.
 *
 * Two functions are compatible if a single local variable can be used to hold
 * their return values and therefore implying that the same return mechanism can
 * be used for the functions.
 *
 */
bool cmpat_func(DTYPE d1, DTYPE d2);

/**
   \brief ...
 */
bool is_array_dtype(DTYPE dtype);

/** Check for special case of empty typedef which has a size of 0
 * but one member of type DT_NONE to indicate that the type is
 * empty and not incomplete, a forward reference, etc.
 */
bool is_empty_typedef(DTYPE dtype);

/** \brief Check for special case of zero-size typedef which may nest have
    zero-size typedef compnents or zero-size array compnents.
 */
bool is_zero_size_typedef(DTYPE dtype);

/**
   \brief ...
 */
bool no_data_components(DTYPE dtype);

bool is_overlap_cmblk_var(int sptr1, int sptr2);

/**
   \brief if array datatype, returns the element dtype, else returns dtype
 */
DTYPE array_element_dtype(DTYPE dtype);

/**
   \brief ...
 */
DTYPE get_array_dtype(int numdim, DTYPE eltype);

/**
   \brief ...
 */
DTYPE get_type(int n, TY_KIND v1, int v2);

/**
   \brief ...
 */
DTYPE get_vector_dtype(DTYPE dtype, int n);

/**
   \brief ...
 */
int alignment(DTYPE dtype);

/**
   \brief ...
 */
int alignment_sym(SPTR sym);

/**
   \brief Support the alignof operator
   \param dtype
 */
int align_of(DTYPE dtype);

/**
   \brief ...
 */
int align_unconstrained(DTYPE dtype);

/**
   \brief Return the length
   \param dty
   Length, in stb.dt_base words, of each type of datatype entry
 */
int dlen(TY_KIND dty);

/**
   \brief ...
 */
int dmp_dent(DTYPE dtypeind);

/**
 * \brief Get FVAL field of a data type
 * \return 0 if reg, 1 if mem.
 */
int fval_of(DTYPE dtype);

/**
   \brief Create a constant sym entry which reflects the type of an array
   bound/extent.
 */
int get_bnd_con(ISZ_T v);

/**
   \brief Extract necessary bytes from character string in order to return
   integer (16-bit) representation of one kanji char.
   \param p the character string
   \param len number of bytes in string p
   \return number of EUC bytes used up
 */
int kanji_char(unsigned char *p, int len, int *bytes);

/**
   \brief Get number of kanji characters
   \param length Length in bytes of character string
 */
int kanji_len(unsigned char *p, int len);

/**
 * \brief Get number of bytes needed for kanji characters in string prefix
 * \param p ptr to EUC string
 * \param newlen number of kanji chars required from string prefix
 * \param len total number of bytes in string
 * \return number of bytes required for newlen chars
 */
int kanji_prefix(unsigned char *p, int newlen, int len);

/**
 * \brief Compute the size of a data type
 * \param dtype
 * \param size    number of elements in the data type [output]
 * \return number of bytes in each element, expressed as a power of two (scale)
 *
 * This machine dependent routine computes the size of a data type in terms of
 * two quantities: the size and the scale
 *
 *  This routine will be used to take advantage of the machines that
 *  have the ability to add a scaled expression (multiplied by a power
 *  of two) to an address.  This is particularly useful for incrementing
 *  a pointer variable and array subscripting.
 *
 *  Note that for those machines that do not have this feature, scale_of
 *  returns a scale of 0 and size_of for size.
 */
int Scale_Of(DTYPE dtype, ISZ_T *size);

/**
   \brief ...
 */
int scale_of(DTYPE dtype, INT *size);

/**
   \brief ...
 */
void dmp_dtype(void);

/**
 * \brief Put into the character array pointed to by ptr, the print
 * representation of dtype.
 */
void getdtype(DTYPE dtype, char *ptr);

/**
   \brief ...
 */
void init_chartab(void);

/**
   \brief ...
 */
void Restore_Chartab(FILE *fil);

/**
   \brief ...
 */
void Save_Chartab(FILE *fil);

int kanji_char(unsigned char*, int, int*);
int kanji_len(unsigned char *, int);

int align_bytes2mask(int bytes);
int align_bytes2power(int bytes);
int align_mask2bytes(int mask);
int align_mask2power(int mask);
int align_power2bytes(int power);
int align_power2mask(int power);

#endif
