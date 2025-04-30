/* Initialize CPU feature data via IFUNC relocation.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.

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

#ifdef SHARED
# include <cpu-features.c>

/* NB: Normally, DL_PLATFORM_INIT calls init_cpu_features to initialize
   CPU features in dynamic executable.  But when loading ld.so inside of
   static executable, DL_PLATFORM_INIT isn't called and IFUNC relocation
   is used to call init_cpu_features.  In static executable, it is called
   once by IFUNC relocation.  In dynamic executable, it is called twice
   by DL_PLATFORM_INIT and by IFUNC relocation.  */
extern void __x86_cpu_features (void) attribute_hidden;
void (*const __x86_cpu_features_p) (void) attribute_hidden
  = __x86_cpu_features;

void
_dl_x86_init_cpu_features (void)
{
  struct cpu_features *cpu_features = __get_cpu_features ();
  if (cpu_features->basic.kind == arch_kind_unknown)
    init_cpu_features (cpu_features);
}

__ifunc (__x86_cpu_features, __x86_cpu_features, NULL, void,
	 _dl_x86_init_cpu_features);
#endif

#undef _dl_x86_get_cpu_features

const struct cpu_features *
_dl_x86_get_cpu_features (void)
{
  return &GLRO(dl_x86_cpu_features);
}
