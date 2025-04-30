/* Common definition for ifunc resolvers.  Linux/ARM version.
   This file is part of the GNU C Library.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

#include <sysdep.h>
#include <ifunc-init.h>

#define INIT_ARCH()

#define arm_libc_ifunc_redirected(redirected_name, name, expr)	\
  __ifunc (redirected_name, name, expr(hwcap), int hwcap, INIT_ARCH)

#if defined SHARED
# define arm_libc_ifunc_hidden_def(redirect_name, name) \
  __hidden_ver1 (name, __GI_##name, redirect_name) \
    __attribute__ ((visibility ("hidden"))) \
    __attribute_copy__ (name)
#else
# define arm_libc_ifunc_hidden_def(redirect_name, name)
#endif
