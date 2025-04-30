/* net/if.h -- declarations for inquiring about network interfaces
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef _NET_IF_H

#define _NET_IF_H	1
#include <features.h>


__BEGIN_DECLS

/* Convert an interface name to an index, and vice versa.  */

extern unsigned int if_nametoindex (const char *__ifname) __THROW;
extern char *if_indextoname (unsigned int __ifindex, char *__ifname) __THROW;

/* Return a list of all interfaces and their indices.  */

struct if_nameindex
  {
    unsigned int if_index;	/* 1, 2, ... */
    char *if_name;		/* null terminated name: "eth0", ... */
  };

extern struct if_nameindex *if_nameindex (void) __THROW;

/* Free the data returned from if_nameindex.  */

extern void if_freenameindex (struct if_nameindex *__ptr) __THROW;

__END_DECLS


#endif /* net/if.h */
