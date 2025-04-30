/* Shared RTM header.  */
#ifndef _HLE_H
#define _HLE_H 1

#include <x86intrin.h>

#define _ABORT_LOCK_BUSY	0xff
#define _ABORT_LOCK_IS_LOCKED	0xfe
#define _ABORT_NESTED_TRYLOCK	0xfd

#endif
