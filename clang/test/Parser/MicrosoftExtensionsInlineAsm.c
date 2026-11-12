// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple i386-mingw32 -fsyntax-only -verify -fms-extensions  %s
// expected-no-diagnostics

void __forceinline InterlockedBitTestAndSet (long *Base, long Bit)
{
  __asm {
    mov eax, Bit
    mov ecx, Base
    lock bts [ecx], eax
    setc al
  };
}
