sinclude(./aclocal.m4)dnl Autoconf lossage
GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/unix/sysv/linux/riscv.

arch_minimum_kernel=4.15.0

libc_cv_riscv_int_abi=no
AC_EGREP_CPP(4 8 8, [__SIZEOF_INT__ __SIZEOF_LONG__ __SIZEOF_POINTER__
  ], libc_cv_riscv_int_abi=lp64)
AC_EGREP_CPP(4 4 4, [__SIZEOF_INT__ __SIZEOF_LONG__ __SIZEOF_POINTER__
  ], libc_cv_riscv_int_abi=ilp32)
if test $libc_cv_riscv_int_abi = no; then
  AC_MSG_ERROR([Unable to determine integer ABI])
fi

if test $libc_cv_riscv_int_abi = ilp32; then
  arch_minimum_kernel=5.4.0
fi

libc_cv_riscv_float_abi=no
AC_EGREP_CPP(yes, [#ifdef __riscv_float_abi_double
		   yes
		   #endif
  ],libc_cv_riscv_float_abi=d)
AC_EGREP_CPP(yes, [#ifdef __riscv_float_abi_soft
		   yes
		   #endif
  ],libc_cv_riscv_float_abi=)
if test "$libc_cv_riscv_float_abi" = no; then
  AC_MSG_ERROR([Unable to determine floating-point ABI])
fi

LIBC_CONFIG_VAR([default-abi], [$libc_cv_riscv_int_abi$libc_cv_riscv_float_abi])

case $libc_cv_riscv_int_abi$libc_cv_riscv_float_abi-$machine in
lp64-riscv/rv64/*)
  LIBC_SLIBDIR_RTLDDIR([lib64/lp64], [lib])
  ;;
lp64d-riscv/rv64/*)
  LIBC_SLIBDIR_RTLDDIR([lib64/lp64d], [lib])
  ;;
ilp32-riscv/rv32/*)
  LIBC_SLIBDIR_RTLDDIR([lib32/ilp32], [lib])
  ;;
ilp32d-riscv/rv32/*)
  LIBC_SLIBDIR_RTLDDIR([lib32/ilp32d], [lib])
  ;;
esac

ldd_rewrite_script=sysdeps/unix/sysv/linux/riscv/ldd-rewrite.sed
