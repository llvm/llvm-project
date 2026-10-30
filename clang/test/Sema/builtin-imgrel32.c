// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -fdeclspec -fsyntax-only -verify=coff %s
// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -fdeclspec -fexperimental-new-constant-interpreter -fsyntax-only -verify=coff %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdeclspec -fsyntax-only -verify=coff %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fdeclspec -fsyntax-only -verify=elf %s

int data;
__declspec(dllimport) int imported;
// elf-warning@-1 {{__declspec attribute 'dllimport' is not supported}}
_Thread_local int tls;

void test(int *p) {
  int local;
  unsigned a = __builtin_imgrel32(&data);
  // elf-error@-1 {{'__builtin_imgrel32' is only supported on COFF targets}}
  unsigned b = __builtin_imgrel32(p);
  // coff-error@-1 {{argument to '__builtin_imgrel32' must be the address of a global variable or function}}
  // elf-error@-2 {{'__builtin_imgrel32' is only supported on COFF targets}}
  unsigned d = __builtin_imgrel32(&local);
  // coff-error@-1 {{argument to '__builtin_imgrel32' must be the address of a global variable or function}}
  // elf-error@-2 {{'__builtin_imgrel32' is only supported on COFF targets}}
  unsigned e = __builtin_imgrel32(&imported);
  // coff-error@-1 {{argument to '__builtin_imgrel32' must be the address of a global variable or function}}
  // elf-error@-2 {{'__builtin_imgrel32' is only supported on COFF targets}}
  unsigned f = __builtin_imgrel32(&tls);
  // coff-error@-1 {{argument to '__builtin_imgrel32' must be the address of a global variable or function}}
  // elf-error@-2 {{'__builtin_imgrel32' is only supported on COFF targets}}
  unsigned c = __builtin_imgrel32();
  // coff-error@-1 {{too few arguments to function call, expected 1, have 0}}
  // elf-error@-2 {{too few arguments to function call, expected 1, have 0}}
}
