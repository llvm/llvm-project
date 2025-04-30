GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.

dnl The standard hppa assembler uses `;' to start comments and `!'
dnl as a line separator.
AC_CACHE_CHECK(for assembler line separator,
	       libc_cv_asm_line_sep, [dnl
cat > conftest.s <<EOF
nop ; is_old_puffin
EOF
if AC_TRY_COMMAND(${CC-cc} -c $ASFLAGS conftest.s 1>&AS_MESSAGE_LOG_FD); then
  libc_cv_asm_line_sep='!'
else
  if test -z "$enable_hacker_mode"; then
    echo "*** You need a newer assembler to compile glibc"
    rm -f conftest*
    exit 1
  fi
  libc_cv_asm_line_sep=';'
fi
rm -f conftest*])
AC_DEFINE_UNQUOTED(ASM_LINE_SEP, $libc_cv_asm_line_sep)
