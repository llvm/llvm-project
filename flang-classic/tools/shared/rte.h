/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file rte.h
 * Run-time data structures
 */

#ifdef RTE_C
/*
 * This section contains declarations which are meant to be visible only
 * to rte.c; if necessary, other modules may enable these declarations by
 * defining RTE_C.
 *
 * The prototypes, declared at the end of this file, represent the interfaces
 * to rte.c; these prototypes are always visible to modules which include
 * this file.
 */

/*
 * The descriptor has two parts: a header and a number of per-dimension
 * parts equal to the array rank.
 * See the file .../rte/hpf/src/pghpf.h for the precise layout of
 * fields within static descriptors.
 */

#define PROC_HDR_INT_LEN 5
#define PROC_DIM_LEN 5

/*
 * for now use just f90 decriptor defs
 */

#define DESC_HDR_INT_LEN 8
#define DESC_HDR_PTR_LEN 2
#define DESC_HDR_LEN                                             \
  (((XBIT(49, 0x100) && !XBIT(49, 0x80000000) && !XBIT(68, 0x1)) \
        ? 2 * DESC_HDR_PTR_LEN                                   \
        : DESC_HDR_PTR_LEN) +                                    \
   DESC_HDR_INT_LEN)

#define DESC_DIM_LEN 6

#define HPF_DESC_HDR_INT_LEN 16
#define HPF_DESC_HDR_PTR_LEN 4
#define HPF_DESC_HDR_LEN                                                  \
  (((XBIT(49, 0x100) && !XBIT(49, 0x80000000)) ? 2 * HPF_DESC_HDR_PTR_LEN \
                                               : HPF_DESC_HDR_PTR_LEN) +  \
   HPF_DESC_HDR_INT_LEN)

#define HPF_DESC_DIM_INT_LEN 34
#define DESC_DIM_PTR_LEN 1
#define HPF_DESC_DIM_LEN                                              \
  (((XBIT(49, 0x100) && !XBIT(49, 0x80000000)) ? 2 * DESC_DIM_PTR_LEN \
                                               : DESC_DIM_PTR_LEN) +  \
   HPF_DESC_DIM_INT_LEN)

/*
 * HPF/F90 common descriptor fields
 */
#define DESC_HDR_TAG 1      /* descriptor tag */
#define DESC_HDR_RANK 2     /* array rank */
#define DESC_HDR_KIND 3     /* array base type */
#define DESC_HDR_BYTE_LEN 4 /* byte length of base type */
#define DESC_HDR_FLAGS 5    /* descriptor flags */
#define DESC_HDR_LSIZE 6    /* local section size */
#define DESC_HDR_GSIZE 7    /* global section size */
#define DESC_HDR_LBASE 8    /* local base index offset */
#define DESC_HDR_GBASE 9    /* base address for debugger */

/* Descriptor flag used to determine if an array section passed as a parameter
 * to an F77 subroutine needs to be copied or whether it can be passed as is.
 * Set in runtime routine ptr_assign and tested by the inline code.
 */
#define __SEQUENTIAL_SECTION 0x20000000
#define __SEQUENTIAL_SECTION_BIT_OFFSET 29
/*
 * HPF specific  descriptor fields
 */

#define HPF_DESC_HDR_HEAPB 2 /* global heap block multiplier */
#define HPF_DESC_HDR_PBASE 3 /* processor base offset */

/*
 * The f77 declaration of a descriptor for an array or section of rank
 * 'r' would be:
 *
 *	 integer*4 a$d(DESC_HDR_LEN + r*DESC_DIM_LEN)
 *
 * Offsets to integer values accessed by the compiler in the header and
 * per-dimension parts should also be defined in the .h file.
 */

/*
 * HPF/F90 Common Dimension descriptor fields
 */
#define DESC_DIM_LOWER 1   /* global lower bound index 	 */
#define DESC_DIM_EXTENT 2  /* global array extent		 */
#define DESC_DIM_SSTRIDE 3 /* section stride 		 */
#define DESC_DIM_SOFFSET 4 /* section offset 		 */
#define DESC_DIM_LMULT 5   /* index linearizing multiplier  */
#define DESC_DIM_UPPER 6   /* global lower bound index  (temporary) */

/*
 * HPF specific dimension descriptor fields
 */

#define HPF_DESC_DIM_LLB 1      /* local lower bound index 	33 */
#define HPF_DESC_DIM_LUB 2      /* local upper bound index 	34 */
#define HFP_DESC_DIM_COFSTR 4   /* cyclic offset stride            */
#define HPF_DESC_DIM_NO 5       /* negative overlap size 	38 */
#define HPF_DESC_DIM_PO 6       /* positive overlap size 	39 */
#define HPF_DESC_DIM_OLB 7      /* owned lower bound 		40 */
#define HPF_DESC_DIM_OUB 8      /* owned upper bound 		41 */
#define HPF_DESC_DIM_TOFFSET 13 /* template offset   		46 */
#define HPF_DESC_DIM_BLOCK 17   /* template block size 		51 */
#define HPF_DESC_DIM_BLOCK_RECIP_HI \
  18 /* block size reciprocal characteristic 52 */
#define HPF_DESC_DIM_BLOCK_RECIP_LO 19 /* block size reciprocal mantissa 53 */
#define HPF_DESC_DIM_PSHAPE 23         /* extent of processor dimension 57 */
#define HPF_DESC_DIM_PSHAPE_RECIP_HI \
  24 /* processor extent reciprocal characteristic 58 */
#define HPF_DESC_DIM_PSHAPE_RECIP_LO \
  25 /* processor extent reciprocal mantissa 59 */
#define HPF_DESC_DIM_PSTRIDE \
  27 /* stride of processor numbers in a dimension 61 */
#define HPF_DESC_DIM_PTR_GENBLOCK            \
  1 /* pointer position of gen-block pointer \
       */

#endif /* RTE_C */

#ifndef __TAGPOLY
#define __TAGPOLY 43
#endif
#ifndef __TAGDESC
#define __TAGDESC 35
#endif
#ifndef __TAGDERIVED
#define __TAGDERIVED 33
#endif

extern int sym_get_proc_sdescr(char *basename, int rank);
extern int sym_get_place_holder(char *basename, int dtype);
extern int sym_get_sdescr(int sptr, int rank);
extern int sym_get_sdescr_inherit(int);
extern void get_static_descriptor(int);
extern void get_all_descriptors(int);
extern int get_multiplier_index(int);
extern int get_global_lower_index(int);
extern int get_global_upper_index(int);
extern int get_global_extent_index(int);
extern int get_section_stride_index(int);
extern int get_section_offset_index(int);
extern int get_local_multiplier(int, int);
extern int get_xbase_index(void);
extern int get_xbase(int);
extern int get_global_lower(int, int);
extern int get_global_upper(int, int);
extern int get_extent(int, int);
extern int get_section_stride(int, int);
extern int get_section_offset(int, int);
extern int get_local_lower(int, int);
extern int get_local_upper(int, int);
extern int get_smp_p2(int);
extern int get_desc_len(void);
extern int get_desc_flags(int);
extern int get_desc_gsize(int);
extern int get_desc_gsize_index(void);
extern int get_desc_lsize(int);
extern int get_lsize_index(void);
extern int get_proc_base(int);
extern int get_proc_stride(int, int);
extern int get_proc_shape(int, int);
extern int get_block_size(int, int);
extern int get_neg_ovlp(int, int);
extern int get_pos_ovlp(int, int);
extern int get_template_offset(int, int);
extern int get_descriptor_len(int rank);
extern int get_owner_lower(int sdsc, int dim);
extern int get_owner_upper(int sdsc, int dim);
extern int get_genblock(int sdsc, int dim);
extern int get_byte_len(int sdsc);
extern int get_byte_len_indx(void);
extern void set_descriptor_sc(int sc);
extern int get_desc_rank(int);
extern int get_kind(int);
extern int get_gbase(int);
extern int get_gbase2(int);
extern int get_desc_tag(int);
extern void rewrite_asn(int, int, bool, int);
extern void get_static_descriptor(int);
extern void set_descriptor_sc(int);
extern int get_descriptor_sc(void);
extern void set_descriptor_rank(int);
extern void set_preserve_descriptor(int);
extern void set_descriptor_class(int);
extern void set_final_descriptor(int);
void set_descriptor_sc(int sc);
int get_header_member_with_parent(int parent, int sdsc, int info);
