GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/powerpc/powerpc32.

# See whether GCC uses -msecure-plt.
AC_CACHE_CHECK(for -msecure-plt by default, libc_cv_ppc_secure_plt, [dnl
echo 'int foo (void) { extern int bar; return bar; }' > conftest.c
libc_cv_ppc_secure_plt=no
if AC_TRY_COMMAND(${CC-cc} -S $CFLAGS conftest.c -fpic -o conftest.s 1>&AS_MESSAGE_LOG_FD); then
  if grep '_GLOBAL_OFFSET_TABLE_-.*@ha' conftest.s > /dev/null 2>&1; then
    libc_cv_ppc_secure_plt=yes
  fi
fi
rm -rf conftest*])
if test $libc_cv_ppc_secure_plt = yes; then
  AC_DEFINE(HAVE_PPC_SECURE_PLT)
fi
