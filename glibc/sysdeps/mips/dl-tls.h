/* Thread-local storage handling in the ELF dynamic linker.  MIPS version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */


/* Type used for the representation of TLS information in the GOT.  */
typedef struct
{
  unsigned long int ti_module;
  unsigned long int ti_offset;
} tls_index;

/* The thread pointer points 0x7000 past the first static TLS block.  */
#define TLS_TP_OFFSET		0x7000

/* Dynamic thread vector pointers point 0x8000 past the start of each
   TLS block.  */
#define TLS_DTV_OFFSET		0x8000

/* Compute the value for a GOTTPREL reloc.  */
#define TLS_TPREL_VALUE(sym_map, sym) \
  ((sym_map)->l_tls_offset + (sym)->st_value - TLS_TP_OFFSET)

/* Compute the value for a DTPREL reloc.  */
#define TLS_DTPREL_VALUE(sym) \
  ((sym)->st_value - TLS_DTV_OFFSET)

extern void *__tls_get_addr (tls_index *ti);

# define GET_ADDR_OFFSET	(ti->ti_offset + TLS_DTV_OFFSET)
# define __TLS_GET_ADDR(__ti)	(__tls_get_addr (__ti) - TLS_DTV_OFFSET)
