dnl We require that everyone use exactly the same Autoconf version so that
dnl the internal functions defined and used by the main configure script
dnl match those expected by the fragments.  When changing this version,
dnl install.texi also needs to be updated.
m4_define([GLIBC_AUTOCONF_VERSION], [2.69])
m4_if(m4_defn([AC_AUTOCONF_VERSION]), GLIBC_AUTOCONF_VERSION, [],
      [m4_fatal(m4_flatten(
Exactly version GLIBC_AUTOCONF_VERSION of Autoconf is required but you have
m4_defn([AC_AUTOCONF_VERSION])
), [63])])dnl
dnl
dnl We define the macro GLIBC_PROVIDES to do an AC_PROVIDE for each macro
dnl which appears in configure.ac before the sysdep configure scripts are run.
dnl Each sysdep configure.ac does GLIBC_PROVIDES first, to avoid any
dnl AC_REQUIREs or AC_BEFOREs duplicating their code.
dnl
define([GLIBC_PROVIDES], [dnl
AC_PROVIDE([AC_CONFIG_AUX_DIR_DEFAULT])dnl
AC_PROVIDE([AC_CONFIG_SUBDIRS])dnl
AC_PROVIDE([_AS_ECHO_N_PREPARE])dnl
AC_PROVIDE([_AS_ECHO_PREPARE])dnl
AC_PROVIDE([_AS_CR_PREPARE])dnl
AC_PROVIDE([_AS_TR_SH_PREPARE])dnl
AC_PROVIDE([_AS_VAR_ARITH_PREPARE])dnl
AC_PROVIDE([AC_PROG_INSTALL])dnl
AC_PROVIDE([AC_PROG_CC])dnl
AC_PROVIDE([AC_PROG_CPP])dnl
AC_PROVIDE([_AS_PATH_SEPARATOR_PREPARE])dnl
AC_PROVIDE([_AS_TEST_PREPARE])dnl
AC_PROVIDE([_AS_BASENAME_PREPARE])dnl
AC_PROVIDE([_AS_ME_PREPARE])dnl
AC_PROVIDE([_AS_LINENO_PREPARE])dnl
AC_PROVIDE([AS_SHELL_FN_as_fn_set_status])dnl
AC_PROVIDE([AS_SHELL_FN_as_fn_exit])dnl
AC_PROVIDE([AS_SHELL_FN_as_fn_arith])dnl
AC_PROVIDE([AS_SHELL_FN_ac_fn_c_try_compile])dnl
AC_PROVIDE([AS_SHELL_FN_ac_fn_c_try_cpp])dnl
define([AS_MESSAGE_LOG_FD],5)dnl
define([AS_MESSAGE_FD],6)dnl
dnl Ripped out of AS_INIT, which does more cruft we do not want.
m4_wrap([m4_divert_pop([BODY])[]])
m4_divert_push([BODY])[]dnl
dnl End of ripped out of AS_INIT.
# This file is generated from configure.ac by Autoconf.  DO NOT EDIT!
define([_AC_LANG], [C])dnl
])dnl
dnl
dnl Check for a symbol
dnl
AC_DEFUN([AC_CHECK_SYMBOL], [dnl
AC_MSG_CHECKING(for $1)
AC_CACHE_VAL(ac_cv_check_symbol_$1, [dnl
AC_TRY_LINK(,
changequote(,)dnl
extern char *$1[]; puts(*$1);,
changequote([,])dnl
	    ac_cv_check_symbol_$1=yes, ac_cv_check_symbol_$1=no)])
if test "$ac_cv_check_symbol_$1" = yes; then
changequote(,)dnl
  ac_tr_symbol=`echo $1 | tr '[a-z]' '[A-Z]'`
changequote([,])dnl
  AC_DEFINE_UNQUOTED(HAVE_${ac_tr_symbol})
fi
AC_MSG_RESULT($ac_cv_check_symbol_$1)])dnl
dnl

dnl Locate a program and check that its version is acceptable.
dnl AC_PROG_CHECK_VER(var, namelist, version-switch,
dnl 		      [version-extract-regexp], version-glob [, do-if-fail])
AC_DEFUN([AC_CHECK_PROG_VER],
[AC_CHECK_PROGS([$1], [$2])
if test -z "[$]$1"; then
  ac_verc_fail=yes
else
  # Found it, now check the version.
  AC_MSG_CHECKING([version of [$]$1])
changequote(<<,>>)dnl
  ac_prog_version=`<<$>>$1 $3 2>&1 ifelse(<<$4>>,,,
		   <<| sed -n 's/^.*patsubst(<<$4>>,/,\/).*$/\1/p'>>)`
  case $ac_prog_version in
    '') ac_prog_version="v. ?.??, bad"; ac_verc_fail=yes;;
    <<$5>>)
changequote([,])dnl
       ac_prog_version="$ac_prog_version, ok"; ac_verc_fail=no;;
    *) ac_prog_version="$ac_prog_version, bad"; ac_verc_fail=yes;;

  esac
  AC_MSG_RESULT([$ac_prog_version])
fi
ifelse([$6],,,
[if test $ac_verc_fail = yes; then
  $6
fi])
])

dnl These modifications are to allow for an empty cross compiler tree.
define([_AC_COMPILER_EXEEXT], [EXEEXT=
])

AC_DEFUN([LIBC_PROG_FOO_GNU],
[# Most GNU programs take a -v and spit out some text including
# the word 'GNU'.  Some try to read stdin, so give them /dev/null.
if $1 -o conftest -v </dev/null 2>&1 | grep GNU > /dev/null 2>&1; then
  $2
else
  $3
fi
rm -fr contest*])

AC_DEFUN([LIBC_PROG_BINUTILS],
[# Was a --with-binutils option given?
if test -n "$path_binutils"; then
    # Make absolute; ensure a single trailing slash.
    path_binutils=`(cd $path_binutils; pwd) | sed 's%/*$%/%'`
    CC="$CC -B$path_binutils"
fi
AS=`$CC -print-prog-name=as`
LD=`$CC -print-prog-name=ld`
AR=`$CC -print-prog-name=ar`
AC_SUBST(AR)
if test "$with_clang" = no; then
    OBJDUMP=`$CC -print-prog-name=objdump`
else
    # llvm-objdump does not support --dynamic-syms
    OBJDUMP=`which objdump`
fi
AC_SUBST(OBJDUMP)
OBJCOPY=`$CC -print-prog-name=objcopy`
AC_SUBST(OBJCOPY)
GPROF=`$CC -print-prog-name=gprof`
AC_SUBST(GPROF)
# When using LTO, llvm-dis must be used search for marker sections in bit-code
# modules in order to list .gnu.glibc-stub.* sections.
if test "$with_next" = yes; then
    LLVM_DIS=`$CC -print-prog-name=llvm-dis`
fi
AC_SUBST(LLVM_DIS)
# Determine whether we are using GNU binutils.
AC_CACHE_CHECK(whether $AS is GNU as, libc_cv_prog_as_gnu,
[LIBC_PROG_FOO_GNU($AS, libc_cv_prog_as_gnu=yes, libc_cv_prog_as_gnu=no)])
rm -f a.out
gnu_as=$libc_cv_prog_as_gnu

AC_CACHE_CHECK(whether $LD is GNU ld, libc_cv_prog_ld_gnu,
[LIBC_PROG_FOO_GNU($LD, libc_cv_prog_ld_gnu=yes, libc_cv_prog_ld_gnu=no)])
gnu_ld=$libc_cv_prog_ld_gnu
])

dnl Run a static link test with -nostdlib -nostartfiles.
dnl LIBC_TRY_LINK_STATIC([code], [action-if-true], [action-if-false])
AC_DEFUN([LIBC_TRY_LINK_STATIC],
[cat > conftest.c <<EOF
int _start (void) { return 0; }
int __start (void) { return 0; }
$1
EOF
AS_IF([AC_TRY_COMMAND([${CC-cc} $CFLAGS $CPPFLAGS $LDFLAGS $no_ssp -o conftest
		       conftest.c -static -nostartfiles -nostdlib
		       1>&AS_MESSAGE_LOG_FD])],
      [$2], [$3])
rm -f conftest*])

dnl Test a compiler option or options with an empty input file.
dnl LIBC_TRY_CC_OPTION([options], [action-if-true], [action-if-false])
AC_DEFUN([LIBC_TRY_CC_OPTION],
[AS_IF([AC_TRY_COMMAND([${CC-cc} $1 -xc /dev/null -S -o /dev/null])],
	[$2], [$3])])

dnl Find and source sysdeps/*/preconfigure.
dnl LIBC_PRECONFIGURE([$srcdir], [for])
AC_DEFUN([LIBC_PRECONFIGURE], [dnl
if frags=`ls -d $1/sysdeps/*/preconfigure 2> /dev/null`
then
  AC_MSG_CHECKING($2 preconfigure fragments)
  for frag in $frags; do
    name=`echo "$frag" | sed 's@/[[^/]]*[$]@@;s@^.*/@@'`
    echo $ECHO_N "$name $ECHO_C" >&AS_MESSAGE_FD
    . "$frag"
  done
  AC_MSG_RESULT()
fi])

# These two macros are taken from GCC's config/acx.m4.
dnl Support the --with-pkgversion configure option.
dnl ACX_PKGVERSION(default-pkgversion)
AC_DEFUN([ACX_PKGVERSION],[
  AC_ARG_WITH(pkgversion,
    AS_HELP_STRING([--with-pkgversion=PKG],
                   [Use PKG in the version string in place of "$1"]),
    [case "$withval" in
      yes) AC_MSG_ERROR([package version not specified]) ;;
      no)  PKGVERSION= ;;
      *)   PKGVERSION="($withval) " ;;
     esac],
    PKGVERSION="($1) "
  )
  PKGVERSION_TEXI=`echo "$PKGVERSION" | sed 's/@/@@/g'`
  AC_SUBST(PKGVERSION)
  AC_SUBST(PKGVERSION_TEXI)
])

dnl Support the --with-bugurl configure option.
dnl ACX_BUGURL(default-bugurl)
AC_DEFUN([ACX_BUGURL],[
  AC_ARG_WITH(bugurl,
    AS_HELP_STRING([--with-bugurl=URL],
                   [Direct users to URL to report a bug]),
    [case "$withval" in
      yes) AC_MSG_ERROR([bug URL not specified]) ;;
      no)  BUGURL=
	   ;;
      *)   BUGURL="$withval"
	   ;;
     esac],
     BUGURL="$1"
  )
  case ${BUGURL} in
  "")
    REPORT_BUGS_TO=
    REPORT_BUGS_TEXI=
    ;;
  *)
    REPORT_BUGS_TO="<$BUGURL>"
    REPORT_BUGS_TEXI=@uref{`echo "$BUGURL" | sed 's/@/@@/g'`}
    ;;
  esac;
  AC_SUBST(REPORT_BUGS_TO)
  AC_SUBST(REPORT_BUGS_TEXI)
])

dnl Check linker option support.
dnl LIBC_LINKER_FEATURE([ld_option], [cc_option], [action-if-true], [action-if-false])
AC_DEFUN([LIBC_LINKER_FEATURE],
[AC_MSG_CHECKING([for linker that supports $1])
libc_linker_feature=no
if test x"$gnu_ld" = x"yes"; then
  libc_linker_check=`$LD -v --help 2>/dev/null | grep "\$1"`
  if test -n "$libc_linker_check"; then
    cat > conftest.c <<EOF
int _start (void) { return 42; }
EOF
    if AC_TRY_COMMAND([${CC-cc} $CFLAGS $CPPFLAGS $LDFLAGS $no_ssp
				$2 -nostdlib -nostartfiles
				-fPIC -shared -o conftest.so conftest.c
				1>&AS_MESSAGE_LOG_FD])
    then
      libc_linker_feature=yes
    fi
    rm -f conftest*
  fi
fi
if test $libc_linker_feature = yes; then
  $3
else
  $4
fi
AC_MSG_RESULT($libc_linker_feature)])

dnl Add a makefile variable, with value set from a shell string
dnl (expanded by the shell inside double quotes), to config.make.
dnl LIBC_CONFIG_VAR(make-variable, shell-value)
AC_DEFUN([LIBC_CONFIG_VAR],
[config_vars="$config_vars
$1 = $2"])

dnl Check that function FUNC was inlined as a builtin.  The code fragment
dnl CODE is compiled with additional options CC_OPTION.  If FUNC is
dnl not found in the assembly then it is assumed the compiler has support
dnl for this builtin and has inlined the call.  If the compiler has the
dnl feature then ACTION-IF-TRUE is called, otherwise ACTION-IF-FALSE.
dnl It is up to the caller to provide a CC_OPTION that ensures the
dnl builtin is inlined if present.
dnl Warning: This may not work for some machines. For example on ARM the
dnl ABI dictates that some functions should not be inlined and instead
dnl should be provided by a compiler helper library e.g. __aeabi_memcpy.
dnl This is done to reduce code size.
dnl LIBC_COMPILER_BUILTIN([func], [code], [cc_option], [action-if-true], [action-if-false])
AC_DEFUN([LIBC_COMPILER_BUILTIN_INLINED],
[AC_MSG_CHECKING([for compiler support of inlined builtin function $1])
libc_compiler_builtin_inlined=no
cat > conftest.c <<EOF
int _start (void) { $2 return 0; }
EOF
if ! AC_TRY_COMMAND([${CC-cc} $CFLAGS $CPPFLAGS $LDFLAGS $no_ssp
		     $3 -nostdlib -nostartfiles
		     -S conftest.c -o - | fgrep "$1"
		     1>&AS_MESSAGE_LOG_FD])
then
  libc_compiler_builtin_inlined=yes
fi
rm -f conftest*
if test $libc_compiler_builtin_inlined = yes; then
  $4
else
  $5
fi
AC_MSG_RESULT($libc_compiler_builtin_inlined)])

dnl Default to slibdir named SLIBDIR instead of "lib", and rtlddir
dnl named RTLDDIR instead of "lib".  This is used to put 64-bit
dnl libraries in /lib64.
dnl LIBC_SLIBDIR_RTLDDIR([slibdir], [rtlddir])
AC_DEFUN([LIBC_SLIBDIR_RTLDDIR],
[test -n "$libc_cv_slibdir" ||
case "$prefix" in
/usr | /usr/)
  libc_cv_slibdir='/$1'
  libc_cv_rtlddir='/$2'
  if test "$libdir" = '${exec_prefix}/lib'; then
    libdir='${exec_prefix}/$1';
    # Locale data can be shared between 32-bit and 64-bit libraries.
    libc_cv_complocaledir='${exec_prefix}/lib/locale'
  fi
  ;;
esac])
