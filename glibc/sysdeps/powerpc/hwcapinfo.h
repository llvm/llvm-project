/* powerpc HWCAP/HWCAP2 and AT_PLATFORM data pre-processing.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <stdint.h>

#ifndef HWCAPINFO_H
# define HWCAPINFO_H

extern uint64_t __tcb_hwcap  attribute_hidden;
extern uint32_t __tcb_platform attribute_hidden;

extern void __tcb_parse_hwcap_and_convert_at_platform (void);

#endif
