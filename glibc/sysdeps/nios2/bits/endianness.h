#ifndef _BITS_ENDIANNESS_H
#define _BITS_ENDIANNESS_H 1

#ifndef _BITS_ENDIAN_H
# error "Never use <bits/endianness.h> directly; include <endian.h> instead."
#endif

/* Nios II has selectable endianness.  */
#ifdef __nios2_big_endian__
# define __BYTE_ORDER __BIG_ENDIAN
#endif
#ifdef __nios2_little_endian__
# define __BYTE_ORDER __LITTLE_ENDIAN
#endif

#endif /* bits/endianness.h */
