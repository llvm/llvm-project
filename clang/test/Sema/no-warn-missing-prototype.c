// RUN: %clang_cc1 -fsyntax-only -Wmissing-prototypes -x c -ffreestanding -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-prototypes -x c++ -ffreestanding -verify %s
// expected-no-diagnostics
int main() {
  return 0;
}

int efi_main() {
  return 0;
}
