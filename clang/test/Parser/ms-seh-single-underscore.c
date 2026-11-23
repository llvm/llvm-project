// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -fsyntax-only -fms-compatibility -Wmicrosoft -verify
// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -x c++ -fsyntax-only -fms-compatibility -Wmicrosoft -verify

int _except(int);

int use_except_identifier(int value) {
  int (*handler)(int) = _except;
  return handler(value) + _except(value);
}

void single_except(void) {
  _try {
    _leave;
  } _except(1) {
  }
}

void single_finally(void) {
  _try {
  } _finally {
  }
}

void mixed_spellings(void) {
  _try {
    __leave;
  } __except(1) {
  }

  __try {
  } _except(1) {
  }

  __try {
  } _finally {
  }
}

void bad_except(void) {
  int value;

  _try {
  } _except(1) value; // expected-error {{expected '{'}} expected-warning {{expression result unused}}
}