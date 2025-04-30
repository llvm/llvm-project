GLIBC_PROVIDES dnl See aclocal.m4 in the top level source directory.
# Local configure fragment for sysdeps/sparc/sparc32

# Test if compiler targets at least sparcv8.
AC_CACHE_CHECK([for at least sparcv8 support],
  [libc_cv_sparcv8],
  [AC_EGREP_CPP(yes,[#if defined (__sparc_v8__) || defined (__sparc_v9__)
                      yes
                     #endif
  ], libc_cv_sparcv8=yes, libc_cv_sparcv8=no)])
if test $libc_cv_sparcv8 = no; then
  AC_MSG_ERROR([no support for pre-v8 sparc])
fi

# Test if compiler generates external calls to libatomic for CAS operation.
# It is suffice to check for int only and the test is similar of C11
# atomic_compare_exchange_strong using GCC builtins.
AC_CACHE_CHECK(for external libatomic calls,
	       libc_cv_cas_uses_libatomic,
	       [cat > conftest.c <<EOF
	       _Bool foo (int *ptr, int *expected, int desired)
	       {
		 return __atomic_compare_exchange_n (ptr, expected, desired, 0,
						     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
	       }
EOF
	       libc_cv_cas_uses_libatomic=no
	       if AC_TRY_COMMAND(${CC-cc} -S conftest.c -o conftest.s 1>&AS_MESSAGE_LOG_FD); then
		 if grep '__atomic_compare_exchange_4' conftest.s >/dev/null; then
		   libc_cv_cas_uses_libatomic=yes
		 fi
	       fi
	       rm -f conftest.c conftest.s
	       ])
if test $libc_cv_cas_uses_libatomic = yes; then
  AC_MSG_ERROR([external dependency of libatomic is not supported])
fi
