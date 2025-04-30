/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <netinet/ether.h>
#include <netinet/if_ether.h>
#include <nss.h>

struct etherent_data {};

#define ENTNAME		etherent
#define DATABASE	"ethers"
#include "files-parse.c"
LINE_PARSER
("#",
 /* Read the ethernet address: 6 x 8bit hexadecimal number.  */
 {
   size_t cnt;

   for (cnt = 0; cnt < 6; ++cnt)
     {
       unsigned int number;

       if (cnt < 5)
	 INT_FIELD (number, ISCOLON , 0, 16, (unsigned int))
       else
	 INT_FIELD (number, isspace, 1, 16, (unsigned int))

       if (number > 0xff)
	 return 0;
       result->e_addr.ether_addr_octet[cnt] = number;
     }
 };
 STRING_FIELD (result->e_name, isspace, 1);
 )


#include GENERIC

DB_LOOKUP (hostton, '.', 0, ("%s", name),
	   {
	     if (__strcasecmp (result->e_name, name) == 0)
	       break;
	   }, const char *name)

DB_LOOKUP (ntohost, '=', 18, ("%x:%x:%x:%x:%x:%x",
			 addr->ether_addr_octet[0], addr->ether_addr_octet[1],
			 addr->ether_addr_octet[2], addr->ether_addr_octet[3],
			 addr->ether_addr_octet[4], addr->ether_addr_octet[5]),
	   {
	     if (memcmp (&result->e_addr, addr,
			 sizeof (struct ether_addr)) == 0)
	       break;
	   }, const struct ether_addr *addr)
