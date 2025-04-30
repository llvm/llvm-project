/* Additional fields in struct link_map.  Linux/x86 version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* if this object has GNU property.  */
enum
  {
    lc_property_unknown = 0,		/* Unknown property status.  */
    lc_property_none = 1 << 0,		/* No property.  */
    lc_property_valid = 1 << 1		/* Has valid property.  */
  } l_property:2;

/* GNU_PROPERTY_X86_FEATURE_1_AND of this object.  */
unsigned int l_x86_feature_1_and;

/* GNU_PROPERTY_X86_ISA_1_NEEDED of this object.  */
unsigned int l_x86_isa_1_needed;
