// RUN: %clang_cc1 -fsyntax-only -Wmissing-prototypes -x c -ffreestanding -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-prototypes -x c++ -ffreestanding -verify %s
// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -Wmissing-prototypes -x c++ -ffreestanding -triple=x86_64-pc-win32 -verify -DMS %s
// expected-no-diagnostics
int main() {
  return 0;
}

int efi_main() {
  return 0;
}

#ifdef MS
int wmain(int, wchar_t *[], wchar_t *[]) {
  return 0;
}

int wWinMain(void*, void*, wchar_t*, int) {
  return 0;
}

int WinMain(void*, void*, char*, int) {
  return 0;
}

bool DllMain(void*, unsigned, void*) {
  return true;
}
#endif
