GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/powerpc/powerpc64.

AC_CACHE_CHECK(for support for overlapping .opd entries,
libc_cv_overlapping_opd, [dnl
libc_cv_overlapping_opd=no
echo 'void foo (void) {}' > conftest.c
if AC_TRY_COMMAND(${CC-cc} $CFLAGS $CPPFLAGS -S conftest.c -o conftest.s 1>&AS_MESSAGE_LOG_FD); then
changequote(,)dnl
  if grep '\.TOC\.@tocbase' conftest.s > /dev/null; then
    if grep '\.TOC\.@tocbase[ 	]*,[ 	]*0' conftest.s > /dev/null; then
      :
    else
      libc_cv_overlapping_opd=yes
    fi
  fi
changequote([,])dnl
fi
rm -f conftest.c conftest.s
])
if test x$libc_cv_overlapping_opd = xyes; then
  AC_DEFINE(USE_PPC64_OVERLAPPING_OPD)
fi

# @notoc started to be supported in GNU Binutils 2.31.
AC_CACHE_CHECK([if the assembler supports @notoc],
	       libc_cv_ppc64_notoc, [
	       AC_COMPILE_IFELSE([AC_LANG_SOURCE([
void foo (void) {asm("b foo@notoc");}
		  ])],
		  [libc_cv_ppc64_notoc=yes],
		  [libc_cv_ppc64_notoc=no])])
AS_IF([test x$libc_cv_ppc64_notoc = xyes],
      [AC_DEFINE(USE_PPC64_NOTOC)])
