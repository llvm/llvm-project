// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only -verify %s

void test_errors(void) {
  unsigned char buf[4];

  __builtin_stdc_memreverse8(4);           // expected-error {{too few arguments to function call}}
  __builtin_stdc_memreverse8(4, buf, buf); // expected-error {{too many arguments to function call}}
}

void test_null_pointer(void) {
  __builtin_stdc_memreverse8(4, 0);                  // expected-warning {{null passed to a callee that requires a non-null argument}}
  __builtin_stdc_memreverse8(4, (unsigned char *)0); // expected-warning {{null passed to a callee that requires a non-null argument}}
}

void test_uintN_pointer_types(void) {
  __UINT8_TYPE__  *p8  = 0;
  __UINT16_TYPE__ *p16 = 0;
  __UINT32_TYPE__ *p32 = 0;
  __UINT64_TYPE__ *p64 = 0;

  __builtin_stdc_memreverse8(4, p8);   // uint8_t* == unsigned char*, no diagnostic
  __builtin_stdc_memreverse8(4, p16);  // expected-error {{incompatible pointer types}}
  __builtin_stdc_memreverse8(4, p32);  // expected-error {{incompatible pointer types}}
  __builtin_stdc_memreverse8(4, p64);  // expected-error {{incompatible pointer types}}
}

void test_pointer_types(void) {
  char *cp = 0;
  int *ip = 0;
  void *vp = 0;
  const unsigned char *ccp = 0;
  signed char *scp = 0;
  unsigned char **ucpp = 0;

  __builtin_stdc_memreverse8(4, cp);   // expected-warning {{converts between pointers to integer types}}
  __builtin_stdc_memreverse8(4, ip);   // expected-error {{incompatible pointer types}}
  __builtin_stdc_memreverse8(4, vp);   // void* implicitly converts, no diagnostic
  __builtin_stdc_memreverse8(4, ccp);  // expected-warning {{discards qualifiers}}
  __builtin_stdc_memreverse8(4, scp);  // expected-warning {{converts between pointers to integer types}}
  __builtin_stdc_memreverse8(4, ucpp); // expected-error {{incompatible pointer types}}
}

#ifdef __has_include
#if __has_include(<stdbit.h>)
#include <stdbit.h>

_Static_assert(stdc_memreverse8u8(0xAB) == 0xAB, "");
_Static_assert(stdc_memreverse8u16(0x1234) == 0x3412, "");
_Static_assert(stdc_memreverse8u32(0x12345678U) == 0x78563412U, "");
_Static_assert(stdc_memreverse8u64(0x123456789ABCDEF0ULL) == 0xF0DEBC9A78563412ULL, "");

_Static_assert(stdc_memreverse8u16(0) == 0, "");
_Static_assert(stdc_memreverse8u32(0) == 0, "");
_Static_assert(stdc_memreverse8u64(0) == 0, "");
_Static_assert(stdc_memreverse8u16(0xFFFF) == 0xFFFF, "");
_Static_assert(stdc_memreverse8u32(0xFFFFFFFFU) == 0xFFFFFFFFU, "");
_Static_assert(stdc_memreverse8u64(0xFFFFFFFFFFFFFFFFULL) == 0xFFFFFFFFFFFFFFFFULL, "");

_Static_assert(stdc_memreverse8u16(stdc_memreverse8u16(0x1234)) == 0x1234, "");
_Static_assert(stdc_memreverse8u32(stdc_memreverse8u32(0x12345678U)) == 0x12345678U, "");
_Static_assert(stdc_memreverse8u64(stdc_memreverse8u64(0x123456789ABCDEF0ULL)) == 0x123456789ABCDEF0ULL, "");
_Static_assert(stdc_memreverse8u8(0xAB) == 0xAB, "");
_Static_assert(stdc_memreverse8u16(0xDEAD) == 0xADDE, "");
_Static_assert(stdc_memreverse8u32(0xDEADBEEFU) == 0xEFBEADDEU, "");
_Static_assert(stdc_memreverse8u64(0x0102030405060708ULL) == 0x0807060504030201ULL, "");

_Static_assert(stdc_memreverse8u8(0xAA) == 0xAA, "");
_Static_assert(stdc_memreverse8u16(0xABAB) == 0xABAB, "");
_Static_assert(stdc_memreverse8u32(0xABCDCDABU) == 0xABCDCDABU, "");
_Static_assert(stdc_memreverse8u64(0x0102030404030201ULL) == 0x0102030404030201ULL, "");

void test_typed_variant_errors(void) {
  stdc_memreverse8u8();         // expected-error {{too few arguments to function call}}
  stdc_memreverse8u8(1, 2);     // expected-error {{too many arguments to function call}}
  stdc_memreverse8u16();        // expected-error {{too few arguments to function call}}
  stdc_memreverse8u16(1, 2);    // expected-error {{too many arguments to function call}}
  stdc_memreverse8u32();        // expected-error {{too few arguments to function call}}
  stdc_memreverse8u32(1, 2);    // expected-error {{too many arguments to function call}}
  stdc_memreverse8u64();        // expected-error {{too few arguments to function call}}
  stdc_memreverse8u64(1, 2);    // expected-error {{too many arguments to function call}}
}
#endif
#endif
