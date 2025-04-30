GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/arm.

dnl It is always possible to access static and hidden symbols in an
dnl position independent way.  This has been true since GCC 4.1,
dnl which is older than the minimum version required to build libc.
AC_DEFINE(PI_STATIC_AND_HIDDEN)

# We check to see if the compiler and flags are
# selecting the hard-float ABI and if they are then
# we set libc_cv_arm_pcs_vfp to yes which causes
# HAVE_ARM_PCS_VFP to be defined in config.h and
# in include/libc-symbols.h and thus available to
# shlib-versions to select the appropriate name for
# the dynamic linker via %ifdef.
AC_CACHE_CHECK([whether the compiler is using the ARM hard-float ABI],
  [libc_cv_arm_pcs_vfp],
  [AC_EGREP_CPP(yes,[#ifdef __ARM_PCS_VFP
		      yes
		     #endif
  ], libc_cv_arm_pcs_vfp=yes, libc_cv_arm_pcs_vfp=no)])
if test $libc_cv_arm_pcs_vfp = yes; then
  AC_DEFINE(HAVE_ARM_PCS_VFP)
  LIBC_CONFIG_VAR([default-abi], [hard])
else
  LIBC_CONFIG_VAR([default-abi], [soft])
fi

AC_CACHE_CHECK([whether PC-relative relocs in movw/movt work properly],
	       libc_cv_arm_pcrel_movw, [
cat > conftest.s <<\EOF
	.syntax unified
	.arm
	.arch armv7-a

	.text
	.globl foo
	.type foo,%function
foo:	movw r0, #:lower16:symbol - 1f - 8
	movt r0, #:upper16:symbol - 1f - 8
1:	add r0, pc
	@ And now a case with a local symbol.
	movw r0, #:lower16:3f - 2f - 8
	movt r0, #:upper16:3f - 2f - 8
2:	add r0, pc
	bx lr

.data
	.globl symbol
	.hidden symbol
symbol:	.long 23
3:	.long 17
EOF
libc_cv_arm_pcrel_movw=no
${CC-cc} $CFLAGS $CPPFLAGS $LDFLAGS \
	 -nostartfiles -nostdlib -shared \
	 -o conftest.so conftest.s 1>&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD &&
LC_ALL=C $READELF -dr conftest.so > conftest.dr 2>&AS_MESSAGE_LOG_FD &&
{
  cat conftest.dr 1>&AS_MESSAGE_LOG_FD
  fgrep 'TEXTREL
R_ARM_NONE' conftest.dr > /dev/null || libc_cv_arm_pcrel_movw=yes
}
rm -f conftest*])
if test $libc_cv_arm_pcrel_movw = yes; then
  AC_DEFINE([ARM_PCREL_MOVW_OK])
fi

# This was buggy in assemblers from GNU binutils versions before 2.25.1
# (it's known to be broken in 2.24 and 2.25; see
# https://sourceware.org/bugzilla/show_bug.cgi?id=18383).
AC_CACHE_CHECK([whether TPOFF relocs with addends are assembled correctly],
	       libc_cv_arm_tpoff_addend, [
cat > conftest.s <<\EOF
	.syntax unified
	.arm
	.arch armv7-a

	.text
foo:
	.word tbase(tpoff)+4

	.section .tdata,"awT",%progbits
	.word -4
tbase:	.word 0
	.word 4
EOF
libc_cv_arm_tpoff_addend=no
${CC-cc} -c $CFLAGS $CPPFLAGS \
	 -o conftest.o conftest.s 1>&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD &&
LC_ALL=C $READELF -x.text conftest.o > conftest.x 2>&AS_MESSAGE_LOG_FD &&
{
  cat conftest.x 1>&AS_MESSAGE_LOG_FD
  $AWK 'BEGIN { result = 2 }
$1 ~ /0x0+/ && $2 ~ /[[0-9a-f]]+/ {
# Check for little-endian or big-endian encoding of 4 in the in-place addend.
  result = ($2 == "04000000" || $2 == "00000004") ? 0 : 1
}
END { exit(result) }
' conftest.x 2>&AS_MESSAGE_LOG_FD && libc_cv_arm_tpoff_addend=yes
}
rm -f conftest*])
if test $libc_cv_arm_tpoff_addend = no; then
  LIBC_CONFIG_VAR([test-xfail-tst-tlsalign], [yes])
  LIBC_CONFIG_VAR([test-xfail-tst-tlsalign-static], [yes])
fi


libc_cv_gcc_unwind_find_fde=no

# Remove -fno-unwind-tables that was added in sysdeps/arm/preconfigure.ac.
CFLAGS=${CFLAGS% -fno-unwind-tables}
