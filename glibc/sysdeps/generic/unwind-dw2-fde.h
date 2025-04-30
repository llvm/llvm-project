/* Subroutines needed for unwinding stack frames for exception handling.  */
/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   Contributed by Jason Merrill <jason@cygnus.com>.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */


struct fde_vector
{
  void *orig_data;
  size_t count;
  struct dwarf_fde *array __flexarr;
};

#ifdef _LIBC
#include <gccframe.h>
#else
struct object
{
  void *pc_begin;
  void *tbase;
  void *dbase;
  union {
    struct dwarf_fde *single;
    struct dwarf_fde **array;
    struct fde_vector *sort;
  } u;

  union {
    struct {
      unsigned long sorted : 1;
      unsigned long from_array : 1;
      unsigned long mixed_encoding : 1;
      unsigned long encoding : 8;
      /* ??? Wish there was an easy way to detect a 64-bit host here;
	 we've got 32 bits left to play with...  */
      unsigned long count : 21;
    } b;
    size_t i;
  } s;

#ifdef DWARF2_OBJECT_END_PTR_EXTENSION
  char *fde_end;
#endif

  struct object *next;
};
#endif

/* This is the original definition of struct object.  While the struct
   itself was opaque to users, they did know how large it was, and
   allocate one statically in crtbegin for each DSO.  Keep this around
   so that we're aware of the static size limitations for the new struct.  */
struct old_object
{
  void *pc_begin;
  void *pc_end;
  struct dwarf_fde *fde_begin;
  struct dwarf_fde **fde_array;
  size_t count;
  struct old_object *next;
};

struct dwarf_eh_bases
{
  void *tbase;
  void *dbase;
  void *func;
};


extern void __register_frame_info_bases (void *, struct object *,
					 void *, void *);
extern void __register_frame_info (void *, struct object *);
extern void __register_frame (void *);
extern void __register_frame_info_table_bases (void *, struct object *,
					       void *, void *);
extern void __register_frame_info_table (void *, struct object *);
extern void __register_frame_table (void *);
extern void *__deregister_frame_info (void *);
extern void *__deregister_frame_info_bases (void *);
extern void __deregister_frame (void *);


typedef          int  sword __attribute__ ((mode (SI)));
typedef unsigned int  uword __attribute__ ((mode (SI)));
typedef unsigned int  uaddr __attribute__ ((mode (pointer)));
typedef          int  saddr __attribute__ ((mode (pointer)));
typedef unsigned char ubyte;

/* Terminology:
   CIE - Common Information Element
   FDE - Frame Descriptor Element

   There is one per function, and it describes where the function code
   is located, and what the register lifetimes and stack layout are
   within the function.

   The data structures are defined in the DWARF specification, although
   not in a very readable way (see LITERATURE).

   Every time an exception is thrown, the code needs to locate the FDE
   for the current function, and starts to look for exception regions
   from that FDE. This works in a two-level search:
   a) in a linear search, find the shared image (i.e. DLL) containing
      the PC
   b) using the FDE table for that shared object, locate the FDE using
      binary search (which requires the sorting).  */

/* The first few fields of a CIE.  The CIE_id field is 0 for a CIE,
   to distinguish it from a valid FDE.  FDEs are aligned to an addressing
   unit boundary, but the fields within are unaligned.  */
struct dwarf_cie
{
  uword length;
  sword CIE_id;
  ubyte version;
  unsigned char augmentation __flexarr;
} __attribute__ ((packed, aligned (__alignof__ (void *))));

/* The first few fields of an FDE.  */
struct dwarf_fde
{
  uword length;
  sword CIE_delta;
  unsigned char pc_begin __flexarr;
} __attribute__ ((packed, aligned (__alignof__ (void *))));

typedef struct dwarf_fde fde;

/* Locate the CIE for a given FDE.  */

static inline struct dwarf_cie *
get_cie (struct dwarf_fde *f)
{
  return (void *)&f->CIE_delta - f->CIE_delta;
}

static inline fde *
next_fde (fde *f)
{
  return (fde *) ((char *) f + f->length + sizeof (f->length));
}

extern fde * _Unwind_Find_FDE (void *, struct dwarf_eh_bases *);

static inline int
last_fde (struct object *obj __attribute__ ((__unused__)), fde *f)
{
#ifdef DWARF2_OBJECT_END_PTR_EXTENSION
  return (char *)f == obj->fde_end || f->length == 0;
#else
  return f->length == 0;
#endif
}
