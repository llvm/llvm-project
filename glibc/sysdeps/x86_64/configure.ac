GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/x86_64.

dnl Check if asm supports AVX512DQ.
AC_CACHE_CHECK(for AVX512DQ support in assembler, libc_cv_asm_avx512dq, [dnl
cat > conftest.s <<\EOF
        vandpd (%rax), %zmm6, %zmm1
EOF
if AC_TRY_COMMAND(${CC-cc} -c $ASFLAGS conftest.s 1>&AS_MESSAGE_LOG_FD); then
  libc_cv_asm_avx512dq=yes
else
  libc_cv_asm_avx512dq=no
fi
rm -f conftest*])
if test $libc_cv_asm_avx512dq = yes; then
  AC_DEFINE(HAVE_AVX512DQ_ASM_SUPPORT)
fi

dnl Check if -mavx512f works.
AC_CACHE_CHECK(for AVX512 support, libc_cv_cc_avx512, [dnl
LIBC_TRY_CC_OPTION([-mavx512f], [libc_cv_cc_avx512=$libc_cv_asm_avx512dq], [libc_cv_cc_avx512=no])
])
if test $libc_cv_cc_avx512 = yes; then
  AC_DEFINE(HAVE_AVX512_SUPPORT)
fi
LIBC_CONFIG_VAR([config-cflags-avx512], [$libc_cv_cc_avx512])

dnl Check if -mprefer-vector-width=128 works.
AC_CACHE_CHECK(-mprefer-vector-width=128, libc_cv_cc_mprefer_vector_width, [dnl
LIBC_TRY_CC_OPTION([-mprefer-vector-width=128],
		   [libc_cv_cc_mprefer_vector_width=yes],
		   [libc_cv_cc_mprefer_vector_width=no])
])
LIBC_CONFIG_VAR([config-cflags-mprefer-vector-width],
		[$libc_cv_cc_mprefer_vector_width])

dnl Check whether asm supports Intel MPX
AC_CACHE_CHECK(for Intel MPX support, libc_cv_asm_mpx, [dnl
cat > conftest.s <<\EOF
        bndmov %bnd0,(%rsp)
EOF
if AC_TRY_COMMAND(${CC-cc} -c $ASFLAGS conftest.s 1>&AS_MESSAGE_LOG_FD); then
  libc_cv_asm_mpx=yes
else
  libc_cv_asm_mpx=no
fi
rm -f conftest*])
if test $libc_cv_asm_mpx = yes; then
  AC_DEFINE(HAVE_MPX_SUPPORT)
fi

if test x"$build_mathvec" = xnotset; then
  build_mathvec=yes
fi

dnl It is always possible to access static and hidden symbols in an
dnl position independent way.
AC_DEFINE(PI_STATIC_AND_HIDDEN)

dnl Static PIE is supported.
AC_DEFINE(SUPPORT_STATIC_PIE)

test -n "$critic_missing" && AC_MSG_ERROR([
*** $critic_missing])
