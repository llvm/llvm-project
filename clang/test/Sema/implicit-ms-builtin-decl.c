// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s -fms-extensions
// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -verify %s -fms-extensions

void f() {
  (void)_byteswap_ushort(42); // expected-warning{{implicitly declaring library function '_byteswap_ushort'}} \
  // expected-note{{include the header <stdlib.h> or explicitly provide a declaration for '_byteswap_ushort'}}
  (void)_byteswap_uint64(42LL); // expected-warning{{implicitly declaring library function '_byteswap_uint64'}} \
  // expected-note{{include the header <stdlib.h> or explicitly provide a declaration for '_byteswap_uint64'}}
}

void _byteswap_ulong(); // expected-warning{{incompatible redeclaration of library function '_byteswap_ulong'}} \
// expected-note{{'_byteswap_ulong' is a builtin}}

unsigned short _byteswap_ushort(unsigned short);
unsigned long long _byteswap_uint64(unsigned long long);

void g() {
  (void)_byteswap_ushort(42);
  (void)_byteswap_uint64(42LL);
}

#if defined(__x86_64__)
void h() {
  (void)__mulh(21, 2);  // expected-warning{{implicitly declaring library function '__mulh'}} \
  // expected-note{{include the header <intrin.h> or explicitly provide a declaration for '__mulh'}}
  (void)__umulh(21, 2); // expected-warning{{implicitly declaring library function '__umulh'}} \
  // expected-note{{include the header <intrin.h> or explicitly provide a declaration for '__umulh'}}
}

long long __mulh(long long, long long);
unsigned long long __umulh(unsigned long long, unsigned long long);

void i() {
  (void)__mulh(21, 2);
  (void)__umulh(21, 2);
}
#endif

#if defined(i386)
void h() {
  (void)__mulh(21LL, 2LL);  // expected-warning{{implicit declaration of function '__mulh' is invalid}}
  (void)__umulh(21ULL, 2ULL);  // expected-warning{{implicit declaration of function '__umulh' is invalid}}
}
#endif
