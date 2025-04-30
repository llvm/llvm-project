/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief F2003 polymorphic/OOP support
 */

#ifndef _TYPE_H_
#define _TYPE_H_

#include "fioMacros.h"

/** \def FUNC(x)
 *  \brief pointer to a procedure macro.
 */
#define FUNC(x) void (*x)()
/** \def VTABLE(x)
 *  \brief pointer to type bound procedure table macro.
 *
 *  This is a pointer to a NULL terminated array of function pointers. 
 */
#define VTABLE(x) void (**x)()

#if defined(TARGET_WIN_X8632)
/** \def FINAL_TABLE(x)
 *  \brief pointer to final procedures table macro.
 *
 * FINAL_TABLE is a pointer to an array of 8 funtion pointers represening
 * final subroutines. Each element represents the recpective rank of the
 * dummy argument.
 */
#define FINAL_TABLE(x) void __stdcall (**x)(char *, char *)
#else
/** \def FINAL_TABLE(x)
 *  \brief pointer to final procedures table macro.
 */
#define FINAL_TABLE(x) void (**x)(char *, char *)
#endif

#define MAX_TYPE_NAME 32

typedef struct object_desc OBJECT_DESC;
typedef struct layout_desc LAYOUT_DESC;
typedef struct proc_desc PROC_DESC;

/** \brief Layout descriptor
 *
 * A layout descriptor will have a list of the following integers representing
 * information on components in the object. For now, we only specify a
 * layout for pointer/allocatable/finalizable components. The layout
 * descriptor is used with sourced allocation (to allocate/clone nested
 * components in an object) and with final subroutines.
 *
 * There are five tags (used in the tag field) to indicate the type of
 * component we're dealing with:
 *
 * - 'T' => an "allocated" pointer (has allocatable attribute)
 * - 'D' => a regular pointer to a derived type (non-recursive)
 * - 'R' => a "recursive" pointer to a derived type
 * - 'P' => a general purpose pointer. Can be a pointer to an array, a
 *        scalar, etc.
 * - 'F' => a non-pointer/non-allocatable finalizable component
 * - 'S' => procedure pointer
 */
struct layout_desc {
  __INT_T tag;                /**< the tag -- one of T,D,R,P,F,S -- see 
                                   description above */
  __INT_T type;               /**< runtime type of component, base type if 
                                   pointer (not yet used) */
  __INT_T offset;             /**< byte offset of component */
  __INT_T length;             /**< length in bytes of component. 0 if unknown */
  __INT_T desc_offset;        /**< byte offset of component's descriptor. -1 if
                                   N/A */
  __INT_T padding;            /**< reserved */
  POINT(TYPE_DESC, declType); /**< ptr to declared type or 0 if N/A  */
};

/** \brief The object_desc structure describes an ``OOP'' F2003 object 
 *
 * NOTE: The fields in object_desc must be consistent in type and length
 * with the fields in F90_Desc minus the F90_DescDim field (see also
 * F90_Desc).
 */
struct object_desc {
   /* Begin overloaded F90_Desc: */
  __INT_T tag;     /**< tag field; usually __POLY (see also _DIST_TYPE) */
  __INT_T baseTag; /**< base tag of variable, usually __DERIVED */
  __INT_T level;   /**< hierarchical position in the inheritance graph */
  __INT_T size;    /**< size of object */
  __INT_T reserved1; /**< reserved */
  __INT_T reserved2; /**< reserved */
  __INT_T reserved3; /**< reserved */
  __INT_T reserved4; /**< reserved */
  POINT(__INT_T, prototype); /**< address of initialized instance */
  POINT(TYPE_DESC, type); /**< address of type of object */
  /* End overloaded F90_Desc */
};

/** \brief The type_desc structure describes the type of an ``OOP'' F2003 
  * object
  * 
  * The type_desc structure "extends" the \ref object_desc structure.
  */
struct type_desc /* extends(OBJECT_DESC) */ {
  OBJECT_DESC obj;             /**< parent object_desc */
  VTABLE(func_table);          /**< pointer to virtual function table */
  POINT(TYPE_DESC, parents);   /**< pointer to parent type descriptor list */
  FINAL_TABLE(finals);         /**< pointer to final procedures table */
  POINT(LAYOUT_DESC, layout);  /**< pointer to layout descriptor */
  char name[MAX_TYPE_NAME + 1];/**< null terminated user defined name of type */
};

/** \brief Procedure Pointer/Argument Descriptor 
  *
  * This descriptor holds the closure pointer of a procedure pointer or
  * procedure dummy argument. This is used to support F2008 internal
  * procedures passed as arguments and pointer targets.
  *
  * NOTE: The fields in object_desc must be consistent in type and length
  * with the fields in F90_Desc minus the F90_DescDim field (see also
  * F90_Desc).
  */
struct proc_desc {
  __INT_T tag;               /**< tag field; usually \ref __PROCPTR 
                                  (see also \ref _DIST_TYPE) */
  __INT_T reserved1;         /**< reserved */
  __INT_T reserved2;         /**< reserved */
  __INT_T reserved3;         /**< reserved */  
  __INT_T reserved4;         /**< reserved */
  __INT_T reserved5;         /**< reserved */
  __INT_T reserved6;         /**< reserved */
  __INT_T reserved7;         /**< reserved */
  POINT(__INT_T, reserved8); /**< reserved */
  POINT(void, closure);      /**< closure/context pointer */
};

extern void I8(__fort_dump_type)(TYPE_DESC *d); /* type.c */

extern __INT_T ENTF90(GET_OBJECT_SIZE, get_object_size)(F90_Desc *d);
extern __INT8_T ENTF90(KGET_OBJECT_SIZE, kget_object_size)(F90_Desc *d);
void ENTF90(SET_TYPE, set_type)(F90_Desc *dd, OBJECT_DESC *td);

#endif
