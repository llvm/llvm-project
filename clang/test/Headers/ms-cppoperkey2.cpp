// RUN: %clang_cc1  -triple x86_64-pc-win32 -fms-compatibility \
// RUN:     -ffreestanding -fsyntax-only -Werror  %s -verify
// RUN: %clang_cc1 \
// RUN:     -ffreestanding -fsyntax-only -Werror  %s -verify
// expected-no-diagnostics
int bb ( int x)
{
  // In user code, treat operator keyword as operator keyword.
  if ( x>1 or x<0) return 1;
  else return 0;  
}
