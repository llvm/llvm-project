// RUN: %clang_cc1 -fsyntax-only -verify %s -fms-extensions

void f() {
  (void)_byteswap_ushort(42); // expected-warning{{implicitly declaring library function '_byteswap_ushort}} \
  // expected-note{{include the header <stdlib.h> or explicitly provide a declaration for '_byteswap_ushort'}}
  (void)_byteswap_uint64(42LL); // expected-warning{{implicitly declaring library function '_byteswap_uint64}} \
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
