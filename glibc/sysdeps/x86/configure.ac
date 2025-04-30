GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/x86.

if test $enable_cet != no; then
  # Check if CET can be enabled.
  AC_CACHE_CHECK(whether CET can be enabled,
		 libc_cv_x86_cet_available, [dnl
cat > conftest.c <<EOF
#if !defined __CET__ || __CET__ != 3
# error CET isn't available.
#endif
EOF
		 if AC_TRY_COMMAND(${CC-cc} -c $CFLAGS -fcf-protection -include cet.h conftest.c 1>&AS_MESSAGE_LOG_FD); then
		   libc_cv_x86_cet_available=yes
		 else
		   libc_cv_x86_cet_available=no
		 fi
		 rm -rf conftest*])
  if test $libc_cv_x86_cet_available != yes; then
    AC_MSG_ERROR([$CC doesn't support CET])
  fi
fi
if test $enable_cet != no; then
  # Check if assembler supports CET.
  AC_CACHE_CHECK(whether $AS supports CET,
		 libc_cv_x86_cet_as, [dnl
cat > conftest.s <<EOF
	incsspd %ecx
EOF
		 if AC_TRY_COMMAND(${CC-cc} -c $CFLAGS conftest.s -o conftest.o 1>&AS_MESSAGE_LOG_FD); then
		   libc_cv_x86_cet_as=yes
		 else
		   libc_cv_x86_cet_as=no
		 fi
		 rm -rf conftest*])
  if test $libc_cv_x86_cet_as = no; then
    AC_MSG_ERROR([$AS doesn't support CET])
  fi
fi
if test $enable_cet = yes; then
  AC_DEFINE(DEFAULT_DL_X86_CET_CONTROL, cet_elf_property)
elif test $enable_cet = permissive; then
  AC_DEFINE(DEFAULT_DL_X86_CET_CONTROL, cet_permissive)
fi
LIBC_CONFIG_VAR([enable-cet], [$enable_cet])

# Check if linker supports x86 ISA level.
AC_CACHE_CHECK([for linker x86 ISA level support],
	       libc_cv_include_x86_isa_level, [dnl
cat > conftest1.S <<EOF
#ifdef __LP64__
# define P2ALIGN 3
#else
# define P2ALIGN 2
#endif
	.section ".note.gnu.property", "a"
	.p2align P2ALIGN
	.long 1f - 0f		/* name length.  */
	.long 4f - 1f		/* data length.  */
	/* NT_GNU_PROPERTY_TYPE_0 */
	.long 5			/* note type.  */
0:
	.asciz "GNU"		/* vendor name.  */
1:
	.p2align P2ALIGN
	/* GNU_PROPERTY_X86_ISA_1_NEEDED */
	.long 0xc0008002	/* pr_type.  */
	.long 3f - 2f		/* pr_datasz.  */
2:
	.long 0x1
3:
	.p2align P2ALIGN
4:
EOF
cat > conftest2.S <<EOF
#ifdef __LP64__
# define P2ALIGN 3
#else
# define P2ALIGN 2
#endif
	.section ".note.gnu.property", "a"
	.p2align P2ALIGN
	.long 1f - 0f		/* name length.  */
	.long 4f - 1f		/* data length.  */
	/* NT_GNU_PROPERTY_TYPE_0 */
	.long 5			/* note type.  */
0:
	.asciz "GNU"		/* vendor name.  */
1:
	.p2align P2ALIGN
	/* GNU_PROPERTY_X86_ISA_1_NEEDED */
	.long 0xc0008002	/* pr_type.  */
	.long 3f - 2f		/* pr_datasz.  */
2:
	.long 0x2
3:
	.p2align P2ALIGN
4:
EOF
libc_cv_include_x86_isa_level=no
libc_cv_have_x86_lahf_sahf=no
libc_cv_have_x86_movbe=no
if AC_TRY_COMMAND(${CC-cc} $CFLAGS $CPPFLAGS -nostartfiles -nostdlib -r -o conftest conftest1.S conftest2.S); then
  count=`LC_ALL=C $READELF -n conftest | grep NT_GNU_PROPERTY_TYPE_0 | wc -l`
  if test "$count" = 1; then
    libc_cv_include_x86_isa_level=yes
    cat > conftest.c <<EOF
EOF
    if AC_TRY_COMMAND(${CC-cc} $CFLAGS $CPPFLAGS -fverbose-asm -S -o - conftest.c) | grep -q "\-msahf"; then
      libc_cv_have_x86_lahf_sahf=yes
    fi
    if AC_TRY_COMMAND(${CC-cc} $CFLAGS $CPPFLAGS -fverbose-asm -S -o - conftest.c) | grep -q "\-mmovbe"; then
      libc_cv_have_x86_movbe=yes
    fi
  fi
fi
rm -f conftest*])
if test $libc_cv_include_x86_isa_level = yes; then
  AC_DEFINE(INCLUDE_X86_ISA_LEVEL)
fi
if test $libc_cv_have_x86_lahf_sahf = yes; then
  AC_DEFINE(HAVE_X86_LAHF_SAHF)
fi
if test $libc_cv_have_x86_movbe = yes; then
  AC_DEFINE(HAVE_X86_MOVBE)
fi
LIBC_CONFIG_VAR([enable-x86-isa-level], [$libc_cv_include_x86_isa_level])
