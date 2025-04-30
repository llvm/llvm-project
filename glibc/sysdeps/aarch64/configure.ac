GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/aarch64.

# Static and hidden objects are accessed without dynamic relocations.
# The exception is -mcmodel=large which is unsupported with PIC/PIE.
AC_DEFINE(PI_STATIC_AND_HIDDEN)

# Static PIE is supported.
AC_DEFINE(SUPPORT_STATIC_PIE)

# We check to see if the compiler and flags are
# selecting the big endian ABI and if they are then
# we set libc_cv_aarch64_be to yes which causes
# HAVE_AARCH64_BE to be defined in config.h and
# in include/libc-symbols.h and thus available to
# shlib-versions to select the appropriate name for
# the dynamic linker via %ifdef.
AC_CACHE_CHECK([for big endian],
  [libc_cv_aarch64_be],
  [AC_EGREP_CPP(yes,[#ifdef __AARCH64EB__
                      yes
                     #endif
  ], libc_cv_aarch64_be=yes, libc_cv_aarch64_be=no)])
if test $libc_cv_aarch64_be = yes; then
  AC_DEFINE(HAVE_AARCH64_BE)
  LIBC_CONFIG_VAR([default-abi], [lp64_be])
else
  LIBC_CONFIG_VAR([default-abi], [lp64])
fi

# Only consider BTI supported if -mbranch-protection=bti is
# on by default in the compiler and the linker produces
# binaries with GNU property notes in PT_GNU_PROPERTY segment.
AC_CACHE_CHECK([for BTI support], [libc_cv_aarch64_bti], [dnl
  cat > conftest.c <<EOF
void foo (void) { }
EOF
  libc_cv_aarch64_bti=no
  if AC_TRY_COMMAND([${CC-cc} $CFLAGS $CPPFLAGS $LDFLAGS -nostdlib -nostartfiles $no_ssp -shared -fPIC -o conftest.so conftest.c]) \
     && AC_TRY_COMMAND([$READELF -lW conftest.so | grep -q GNU_PROPERTY]) \
     && AC_TRY_COMMAND([$READELF -nW conftest.so | grep -q "NT_GNU_PROPERTY_TYPE_0.*AArch64 feature:.* BTI"])
  then
    libc_cv_aarch64_bti=yes
  fi
  rm -rf conftest.*])
LIBC_CONFIG_VAR([aarch64-bti], [$libc_cv_aarch64_bti])
if test $libc_cv_aarch64_bti = yes; then
  AC_DEFINE(HAVE_AARCH64_BTI)
fi

# Check if glibc is built with return address signing, i.e.
# if -mbranch-protection=pac-ret is on. We need this because
# pac-ret relies on unwinder support so it's not safe to use
# it in assembly code unconditionally, but there is no
# feature test macro for it in gcc.
AC_CACHE_CHECK([if pac-ret is enabled], [libc_cv_aarch64_pac_ret], [dnl
  cat > conftest.c <<EOF
int bar (void);
int foo (void) { return bar () + 1; }
EOF
  libc_cv_aarch64_pac_ret=no
  if AC_TRY_COMMAND([${CC-cc} $CFLAGS $CPPFLAGS -S -o conftest.s conftest.c]) \
     && AC_TRY_COMMAND([grep -q -E '\''(hint( |	)+25|paciasp)'\'' conftest.s])
  then
    libc_cv_aarch64_pac_ret=yes
  fi
  rm -rf conftest.*])
if test $libc_cv_aarch64_pac_ret = yes; then
  AC_DEFINE(HAVE_AARCH64_PAC_RET)
fi

# Check if binutils supports variant PCS symbols.
AC_CACHE_CHECK([for variant PCS support], [libc_cv_aarch64_variant_pcs], [dnl
  cat > conftest.S <<EOF
.global foo
.type foo, %function
.variant_pcs foo
foo:
	ret
.global bar
.type bar, %function
bar:
	b foo
EOF
  libc_cv_aarch64_variant_pcs=no
  if AC_TRY_COMMAND([${CC-cc} $CFLAGS $CPPFLAGS $LDFLAGS -nostdlib -nostartfiles $no_ssp -shared -fPIC -o conftest.so conftest.S]) \
     && AC_TRY_COMMAND([$READELF -dW conftest.so | grep -q AARCH64_VARIANT_PCS])
  then
    libc_cv_aarch64_variant_pcs=yes
  fi
  rm -rf conftest.*])
LIBC_CONFIG_VAR([aarch64-variant-pcs], [$libc_cv_aarch64_variant_pcs])

# Check if asm support armv8.2-a+sve
AC_CACHE_CHECK(for SVE support in assembler, libc_cv_asm_sve, [dnl
cat > conftest.s <<\EOF
        ptrue p0.b
EOF
if AC_TRY_COMMAND(${CC-cc} -c -march=armv8.2-a+sve conftest.s 1>&AS_MESSAGE_LOG_FD); then
  libc_cv_aarch64_sve_asm=yes
else
  libc_cv_aarch64_sve_asm=no
fi
rm -f conftest*])
if test $libc_cv_aarch64_sve_asm = yes; then
  AC_DEFINE(HAVE_AARCH64_SVE_ASM)
fi
