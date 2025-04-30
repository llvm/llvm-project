GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/unix/sysv/linux/powerpc/powerpc64/.

LIBC_SLIBDIR_RTLDDIR([lib64], [lib64])

# Define default-abi according to compiler flags.
AC_CACHE_CHECK([whether the compiler is using the PowerPC64 ELFv2 ABI],
  [libc_cv_ppc64_elfv2_abi],
  [AC_EGREP_CPP(yes,[#if _CALL_ELF == 2
                      yes
                     #endif
  ], libc_cv_ppc64_elfv2_abi=yes, libc_cv_ppc64_elfv2_abi=no)])
if test $libc_cv_ppc64_elfv2_abi = yes; then
  LIBC_CONFIG_VAR([default-abi], [64-v2])
  # For shlib-versions.
  AC_DEFINE(HAVE_ELFV2_ABI)
else
  LIBC_CONFIG_VAR([default-abi], [64-v1])
  # Compiler that do not support ELFv2 ABI does not define _CALL_ELF
  AC_CACHE_CHECK([whether the compiler defines _CALL_ELF],
    [libc_cv_ppc64_def_call_elf],
    [AC_EGREP_CPP(yes,[#ifdef _CALL_ELF
                         yes
                       #endif
    ], libc_cv_ppc64_def_call_elf=yes, libc_cv_ppc64_def_call_elf=no)])
  if test $libc_cv_ppc64_def_call_elf = no; then
    libc_extra_cppflags="$libc_extra_cppflags -D_CALL_ELF=1"
  fi
fi
# Set minimum kernel version for ppc64le.
AC_CACHE_CHECK([the machine type to set minimum kernel version],
  [libc_cv_ppc64_le],
  [AC_EGREP_CPP(yes,[#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                      yes
                     #endif
  ], arch_minimum_kernel=3.10.0, libc_cv_ppc64_le=no)])
