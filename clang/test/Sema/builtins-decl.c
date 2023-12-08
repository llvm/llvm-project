// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=i686-mingw32
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=x86_64-mingw32
// expected-no-diagnostics

// mingw-w64's intrin.h has decls below.
// we should accept them.
extern unsigned int __builtin_ia32_crc32qi (unsigned int, unsigned char);
extern unsigned int __builtin_ia32_crc32hi (unsigned int, unsigned short);
extern unsigned int __builtin_ia32_crc32si (unsigned int, unsigned int);

// GCC documents these as unsigned, but they are defined with a signed argument.
extern int __builtin_ffs(int);
extern int __builtin_ffsl(long);
extern int __builtin_ffsll(long long);
