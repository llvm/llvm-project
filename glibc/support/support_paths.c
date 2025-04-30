/* Various paths that might be needed.
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

#include <support/support.h>
#include <support/check.h>

/* The idea here is to make various makefile-level paths available to
   support programs, as canonicalized absolute paths.  */

/* These point to the TOP of the source/build tree, not your (or
   support's) subdirectory.  */
#ifdef SRCDIR_PATH
const char support_srcdir_root[] = SRCDIR_PATH;
#else
# error please -DSRCDIR_PATH=something in the Makefile
#endif

#ifdef OBJDIR_PATH
const char support_objdir_root[] = OBJDIR_PATH;
#else
# error please -DOBJDIR_PATH=something in the Makefile
#endif

#ifdef OBJDIR_ELF_LDSO_PATH
/* Corresponds to the path to the runtime linker used by the testsuite,
   e.g. OBJDIR_PATH/elf/ld-linux-x86-64.so.2  */
const char support_objdir_elf_ldso[] = OBJDIR_ELF_LDSO_PATH;
#else
# error please -DOBJDIR_ELF_LDSO_PATH=something in the Makefile
#endif

#ifdef INSTDIR_PATH
/* Corresponds to the --prefix= passed to configure.  */
const char support_install_prefix[] = INSTDIR_PATH;
#else
# error please -DINSTDIR_PATH=something in the Makefile
#endif

#ifdef LIBDIR_PATH
/* Corresponds to the install's lib/ or lib64/ directory.  */
const char support_libdir_prefix[] = LIBDIR_PATH;
#else
# error please -DLIBDIR_PATH=something in the Makefile
#endif

#ifdef BINDIR_PATH
/* Corresponds to the install's bin/ directory.  */
const char support_bindir_prefix[] = BINDIR_PATH;
#else
# error please -DBINDIR_PATH=something in the Makefile
#endif

#ifdef SBINDIR_PATH
/* Corresponds to the install's bin/ directory.  */
const char support_sbindir_prefix[] = SBINDIR_PATH;
#else
# error please -DSBINDIR_PATH=something in the Makefile
#endif

#ifdef SLIBDIR_PATH
/* Corresponds to the system /lib or /lib64 directory.  */
const char support_slibdir_prefix[] = SLIBDIR_PATH;
#else
# error please -DSLIBDIR_PATH=something in the Makefile
#endif

#ifdef ROOTSBINDIR_PATH
/* Corresponds to the install's sbin/ directory.  */
const char support_install_rootsbindir[] = ROOTSBINDIR_PATH;
#else
# error please -DROOTSBINDIR_PATH=something in the Makefile
#endif

#ifdef COMPLOCALEDIR_PATH
/* Corresponds to the install's compiled locale directory.  */
const char support_complocaledir_prefix[] = COMPLOCALEDIR_PATH;
#else
# error please -DCOMPLOCALEDIR_PATH=something in the Makefile
#endif
