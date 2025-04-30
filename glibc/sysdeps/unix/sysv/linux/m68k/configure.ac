sinclude(./aclocal.m4)dnl Autoconf lossage
GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
case $machine in
m68k/coldfire)
  ;;
*)
  libc_cv_gcc_unwind_find_fde=yes
  ;;
esac
ldd_rewrite_script=sysdeps/unix/sysv/linux/ldd-rewrite.sed
