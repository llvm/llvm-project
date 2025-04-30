/* Unit test for _dl_addr_inside_object.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <link.h>
#include <elf.h>
#include <libc-symbols.h>

extern int _dl_addr_inside_object (struct link_map *l, const ElfW(Addr) addr);

static int
do_test (void)
{
  int ret, err = 0;
  ElfW(Addr) addr;
  struct link_map map;
  ElfW(Phdr) header;
  map.l_phdr = &header;
  map.l_phnum = 1;
  map.l_addr = 0x0;
  /* Segment spans 0x2000 -> 0x4000.  */
  header.p_vaddr = 0x2000;
  header.p_memsz = 0x2000;
  header.p_type = PT_LOAD;
  /* Address is above the segment e.g. > 0x4000.  */
  addr = 0x5000;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("PASS: Above: Address is detected as outside the segment.\n");
	break;
      case 1:
        printf ("FAIL: Above: Address is detected as inside the segment.\n");
	err++;
	break;
      default:
	printf ("FAIL: Above: Invalid return value.\n");
	exit (1);
    }
  /* Address is inside the segment e.g. 0x2000 < addr < 0x4000.  */
  addr = 0x3000;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("FAIL: Inside: Address is detected as outside the segment.\n");
	err++;
	break;
      case 1:
        printf ("PASS: Inside: Address is detected as inside the segment.\n");
	break;
      default:
	printf ("FAIL: Inside: Invalid return value.\n");
	exit (1);
    }
  /* Address is below the segment e.g. < 0x2000.  */
  addr = 0x1000;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("PASS: Below: Address is detected as outside the segment.\n");
	break;
      case 1:
        printf ("FAIL: Below: Address is detected as inside the segment.\n");
	err++;
	break;
      default:
	printf ("FAIL: Below: Invalid return value.\n");
	exit (1);
    }
  /* Address is in the segment and addr == p_vaddr.  */
  addr = 0x2000;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("FAIL: At p_vaddr: Address is detected as outside the segment.\n");
	err++;
	break;
      case 1:
        printf ("PASS: At p_vaddr: Address is detected as inside the segment.\n");
	break;
      default:
	printf ("FAIL: At p_vaddr: Invalid return value.\n");
	exit (1);
    }
  /* Address is in the segment and addr == p_vaddr + p_memsz - 1.  */
  addr = 0x2000 + 0x2000 - 0x1;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("FAIL: At p_memsz-1: Address is detected as outside the segment.\n");
	err++;
	break;
      case 1:
        printf ("PASS: At p_memsz-1: Address is detected as inside the segment.\n");
	break;
      default:
	printf ("FAIL: At p_memsz-1: Invalid return value.\n");
	exit (1);
    }
  /* Address is outside the segment and addr == p_vaddr + p_memsz.  */
  addr = 0x2000 + 0x2000;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("PASS: At p_memsz: Address is detected as outside the segment.\n");
	break;
      case 1:
        printf ("FAIL: At p_memsz: Address is detected as inside the segment.\n");
	err++;
	break;
      default:
	printf ("FAIL: At p_memsz: Invalid return value.\n");
	exit (1);
    }
  /* Address is outside the segment and p_vaddr at maximum address.  */
  addr = 0x0 - 0x2;
  header.p_vaddr = 0x0 - 0x1;
  header.p_memsz = 0x1;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("PASS: At max: Address is detected as outside the segment.\n");
	break;
      case 1:
        printf ("FAIL: At max: Address is detected as inside the segment.\n");
	err++;
	break;
      default:
	printf ("FAIL: At max: Invalid return value.\n");
	exit (1);
    }
  /* Address is outside the segment and p_vaddr at minimum address.  */
  addr = 0x1;
  header.p_vaddr = 0x0;
  header.p_memsz = 0x1;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("PASS: At min: Address is detected as outside the segment.\n");
	break;
      case 1:
        printf ("FAIL: At min: Address is detected as inside the segment.\n");
	err++;
	break;
      default:
	printf ("FAIL: At min: Invalid return value.\n");
	exit (1);
    }
  /* Address is always inside the segment with p_memsz at max.  */
  addr = 0x0;
  header.p_vaddr = 0x0;
  header.p_memsz = 0x0 - 0x1;
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("FAIL: At maxmem: Address is detected as outside the segment.\n");
	err++;
	break;
      case 1:
        printf ("PASS: At maxmem: Address is detected as inside the segment.\n");
	break;
      default:
	printf ("FAIL: At maxmem: Invalid return value.\n");
	exit (1);
    }
  /* Attempt to wrap addr into the segment.
     Pick a load address in the middle of the address space.
     Place the test address at 0x0 so it wraps to the middle again.  */
  map.l_addr = 0x0 - 0x1;
  map.l_addr = map.l_addr / 2;
  addr = 0;
  /* Setup a segment covering 1/2 the address space.  */
  header.p_vaddr = 0x0;
  header.p_memsz = 0x0 - 0x1 - map.l_addr;
  /* No matter where you place addr everything is shifted modulo l_addr
     and even with this underflow you're always 1 byte away from being
     in the range.  */
  ret = _dl_addr_inside_object (&map, addr);
  switch (ret)
    {
      case 0:
	printf ("PASS: Underflow: Address is detected as outside the segment.\n");
	break;
      case 1:
	printf ("FAIL: Underflow: Address is detected as inside the segment.\n");
	err++;
	break;
      default:
	printf ("FAIL: Underflow: Invalid return value.\n");
	exit (1);
    }

  return err;
}

#include <support/test-driver.c>
