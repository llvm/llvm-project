// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -fsyntax-only -fms-extensions -verify
// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -x c++ -fsyntax-only -fms-extensions -verify

int _except(int);

int use_except_identifier(int value) {
  return _except(value);
}

void double_except(void) {
  __try {
  } __except(1) {
  }
}

void single_except(void) {
  __try {
  } _except(1) { // expected-error {{expected '__except' or '__finally' block}} expected-error {{expected ';' after expression}}
  }
}