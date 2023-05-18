// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions -triple i386-pc-win32 %s

template <typename T>
int wmain() { // expected-error{{'wmain' cannot be a template}}
  return 0;
}

namespace {
int WinMain(void) { return 0; }
int WinMain(int) { return 0; }
}

void wWinMain(void) {} // expected-note{{previous definition is here}}
void wWinMain(int) {} // expected-error{{conflicting types for 'wWinMain'}}

int foo() {
  wmain<void>(); // expected-error{{no matching function for call to 'wmain'}}
  wmain<int>(); // expected-error{{no matching function for call to 'wmain'}}
  WinMain();
  return 0;
}
