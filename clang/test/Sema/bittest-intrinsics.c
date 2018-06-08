// RUN: %clang_cc1 -ffreestanding -fms-compatibility -fms-compatibility-version=19 -verify %s -triple i686-windows-msvc -fms-extensions -DTEST_X86
// RUN: %clang_cc1 -ffreestanding -fms-compatibility -fms-compatibility-version=19 -verify %s -triple x86_64-windows-msvc -fms-extensions -DTEST_X64
// RUN: %clang_cc1 -ffreestanding -fms-compatibility -fms-compatibility-version=19 -verify %s -triple arm-windows-msvc -fms-extensions -DTEST_ARM
// RUN: %clang_cc1 -ffreestanding -fms-compatibility -fms-compatibility-version=19 -verify %s -triple thumbv7-windows-msvc -fms-extensions -DTEST_ARM
// RUN: %clang_cc1 -ffreestanding -fms-compatibility -fms-compatibility-version=19 -verify %s -triple aarch64-windows-msvc -fms-extensions -DTEST_ARM

#include <intrin.h>
extern unsigned char sink;

#ifdef TEST_X86
void x86(long *bits, __int64 *bits64, long bitidx) {
  sink = _bittest(bits, bitidx);
  sink = _bittestandcomplement(bits, bitidx);
  sink = _bittestandreset(bits, bitidx);
  sink = _bittestandset(bits, bitidx);
  sink = _interlockedbittestandreset(bits, bitidx);
  sink = _interlockedbittestandset(bits, bitidx);

  sink = _bittest64(bits64, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _bittestandcomplement64(bits64, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _bittestandreset64(bits64, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _bittestandset64(bits64, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandreset64(bits64, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandset64(bits64, bitidx); // expected-error {{builtin is not supported on this target}}

  sink = _interlockedbittestandreset_acq(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandreset_rel(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandreset_nf(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandset_acq(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandset_rel(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandset_nf(bits, bitidx); // expected-error {{builtin is not supported on this target}}
}
#endif

#ifdef TEST_X64
void x64(long *bits, __int64 *bits64, long bitidx) {
  sink = _bittest(bits, bitidx);
  sink = _bittestandcomplement(bits, bitidx);
  sink = _bittestandreset(bits, bitidx);
  sink = _bittestandset(bits, bitidx);
  sink = _interlockedbittestandreset(bits, bitidx);
  sink = _interlockedbittestandset(bits, bitidx);

  sink = _bittest64(bits64, bitidx);
  sink = _bittestandcomplement64(bits64, bitidx);
  sink = _bittestandreset64(bits64, bitidx);
  sink = _bittestandset64(bits64, bitidx);
  sink = _interlockedbittestandreset64(bits64, bitidx);
  sink = _interlockedbittestandset64(bits64, bitidx);

  sink = _interlockedbittestandreset_acq(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandreset_rel(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandreset_nf(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandset_acq(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandset_rel(bits, bitidx); // expected-error {{builtin is not supported on this target}}
  sink = _interlockedbittestandset_nf(bits, bitidx); // expected-error {{builtin is not supported on this target}}
}
#endif

#ifdef TEST_ARM
// expected-no-diagnostics
void arm(long *bits, __int64 *bits64, long bitidx) {
  sink = _bittest(bits, bitidx);
  sink = _bittestandcomplement(bits, bitidx);
  sink = _bittestandreset(bits, bitidx);
  sink = _bittestandset(bits, bitidx);
  sink = _interlockedbittestandreset(bits, bitidx);
  sink = _interlockedbittestandset(bits, bitidx);

  sink = _bittest64(bits64, bitidx);
  sink = _bittestandcomplement64(bits64, bitidx);
  sink = _bittestandreset64(bits64, bitidx);
  sink = _bittestandset64(bits64, bitidx);
  sink = _interlockedbittestandreset64(bits64, bitidx);
  sink = _interlockedbittestandset64(bits64, bitidx);

  sink = _interlockedbittestandreset_acq(bits, bitidx);
  sink = _interlockedbittestandreset_rel(bits, bitidx);
  sink = _interlockedbittestandreset_nf(bits, bitidx);
  sink = _interlockedbittestandset_acq(bits, bitidx);
  sink = _interlockedbittestandset_rel(bits, bitidx);
  sink = _interlockedbittestandset_nf(bits, bitidx);
}
#endif
