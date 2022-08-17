// RUN: %clang_cc1 -triple x86_64-unknown-linux -verify %s
// RUN: %clang_cc1 -triple i686-unknown-linux -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-win32 -verify %s
// RUN: %clang_cc1 -triple i686-unknown-win32 -verify %s

/* WG14 N629: yes
 * integer constant type rules
 */

// expected-no-diagnostics

void test_decimal_constants(void) {
  // Easy cases where the value fits into the type you'd expect.
  (void)_Generic(2,    int : 1);
  (void)_Generic(2u,   unsigned int : 1);
  (void)_Generic(2l,   long : 1);
  (void)_Generic(2ul,  unsigned long : 1);
  (void)_Generic(2ll,  long long : 1);
  (void)_Generic(2ull, unsigned long long : 1);

#if __INT_WIDTH__ == 16
  #if __LONG_WIDTH__ > 16
    (void)_Generic(65536, long : 1);
    (void)_Generic(65536U, unsigned long : 1);
  #else
    (void)_Generic(65536, long long : 1);
    (void)_Generic(65536U, unsigned long : 1);
  #endif // __LONG_WIDTH__ > 16
#elif __INT_WIDTH__ == 32
  #if __LONG_WIDTH__ > 32
    (void)_Generic(4294967296, long : 1);
    (void)_Generic(4294967296U, unsigned long : 1);
  #else
    (void)_Generic(4294967296, long long : 1);
    (void)_Generic(4294967296U, unsigned long long : 1);
  #endif // __LONG_WIDTH__ > 32
#endif

#if __LONG_WIDTH__ > 32
  (void)_Generic(4294967296L, long : 1);
  (void)_Generic(4294967296U, unsigned long : 1);
#else
  (void)_Generic(4294967296L, long long : 1);
  (void)_Generic(4294967296U, unsigned long long : 1);
#endif
}

void test_octal_constants(void) {
  (void)_Generic(02,    int : 1);
  (void)_Generic(02u,   unsigned int : 1);
  (void)_Generic(02l,   long : 1);
  (void)_Generic(02ul,  unsigned long : 1);
  (void)_Generic(02ll,  long long : 1);
  (void)_Generic(02ull, unsigned long long : 1);

#if __INT_WIDTH__ == 16
  #if __LONG_WIDTH__ > 16
    (void)_Generic(0200000, long : 1);
    (void)_Generic(0200000U, unsigned long : 1);
  #else
    (void)_Generic(0200000, long long : 1);
    (void)_Generic(0200000U, unsigned long : 1);
  #endif // __LONG_WIDTH__ > 16
#elif __INT_WIDTH__ == 32
  #if __LONG_WIDTH__ > 32
    (void)_Generic(040000000000, long : 1);
    (void)_Generic(040000000000U, unsigned long : 1);
  #else
    (void)_Generic(040000000000, long long : 1);
    (void)_Generic(040000000000U, unsigned long long : 1);
  #endif // __LONG_WIDTH__ > 32
#endif

#if __LONG_WIDTH__ > 32
  (void)_Generic(040000000000L, long : 1);
  (void)_Generic(040000000000U, unsigned long : 1);
#else
  (void)_Generic(040000000000L, long long : 1);
  (void)_Generic(040000000000U, unsigned long long : 1);
#endif
}

void test_hexadecimal_constants(void) {
  (void)_Generic(0x2,    int : 1);
  (void)_Generic(0x2u,   unsigned int : 1);
  (void)_Generic(0x2l,   long : 1);
  (void)_Generic(0x2ul,  unsigned long : 1);
  (void)_Generic(0x2ll,  long long : 1);
  (void)_Generic(0x2ull, unsigned long long : 1);

#if __INT_WIDTH__ == 16
  #if __LONG_WIDTH__ > 16
    (void)_Generic(0x10000, long : 1);
    (void)_Generic(0x10000U, unsigned long : 1);
  #else
    (void)_Generic(0x10000, long long : 1);
    (void)_Generic(0x10000U, unsigned long : 1);
  #endif // __LONG_WIDTH__ > 16
#elif __INT_WIDTH__ == 32
  #if __LONG_WIDTH__ > 32
    (void)_Generic(0x100000000, long : 1);
    (void)_Generic(0x100000000U, unsigned long : 1);
  #else
    (void)_Generic(0x100000000, long long : 1);
    (void)_Generic(0x100000000U, unsigned long long : 1);
  #endif // __LONG_WIDTH__ > 32
#endif

#if __LONG_WIDTH__ > 32
  (void)_Generic(0x100000000L, long : 1);
  (void)_Generic(0x100000000U, unsigned long : 1);
#else
  (void)_Generic(0x100000000L, long long : 1);
  (void)_Generic(0x100000000U, unsigned long long : 1);
#endif
}
