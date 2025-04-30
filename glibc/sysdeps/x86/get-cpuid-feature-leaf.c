/* Get CPUID feature leaf.
   Copyright (C) 2021 Free Software Foundation, Inc.

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


#include <ldsodefs.h>

const struct cpuid_feature *
__x86_get_cpuid_feature_leaf (unsigned int leaf)
{
  static const struct cpuid_feature feature = {};
  if (leaf < CPUID_INDEX_MAX)
    return ((const struct cpuid_feature *)
	      &GLRO(dl_x86_cpu_features).features[leaf]);
  else
    return &feature;
}
