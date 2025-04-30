#define ftw64 __rename_ftw64
#define nftw64 __rename_nftw64

#include "../../io/ftw.c"

#undef ftw64
#undef nftw64

weak_alias (ftw, ftw64)
strong_alias (__new_nftw, __new_nftw64)
versioned_symbol (libc, __new_nftw64, nftw64, GLIBC_2_3_3);

#if SHLIB_COMPAT(libc, GLIBC_2_1, GLIBC_2_3_3)
strong_alias (__old_nftw, __old_nftw64)
compat_symbol (libc, __old_nftw64, nftw64, GLIBC_2_1);
#endif
