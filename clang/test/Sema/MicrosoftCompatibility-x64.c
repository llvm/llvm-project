// RUN: %clang_cc1 %s -Wmicrosoft -verify -fms-compatibility -triple x86_64-pc-win32

// None of these should warn. stdcall is treated as equivalent to cdecl on
// x64.
// expected-no-diagnostics

int __stdcall f(void);
int __cdecl f(void) {
  return 0;
}
int __stdcall func_std(void);
int __thiscall func_this(void);
int __fastcall func_fast(void);
