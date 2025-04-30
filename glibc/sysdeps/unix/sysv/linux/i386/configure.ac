GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/unix/sysv/linux/i386.

# Check if CFLAGS allows compiler to use ebp register in inline assembly.
AC_CACHE_CHECK([if compiler flags allows ebp in inline assembly],
                libc_cv_can_use_register_asm_ebp, [
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([
    void foo (int i)
    {
      register int reg asm ("ebp") = i;
      asm ("# %0" : : "r" (reg));
    }])],
  [libc_cv_can_use_register_asm_ebp=yes],
  [libc_cv_can_use_register_asm_ebp=no])
])
if test $libc_cv_can_use_register_asm_ebp = yes; then
  AC_DEFINE(CAN_USE_REGISTER_ASM_EBP)
fi

libc_cv_gcc_unwind_find_fde=yes
ldd_rewrite_script=sysdeps/unix/sysv/linux/ldd-rewrite.sed
