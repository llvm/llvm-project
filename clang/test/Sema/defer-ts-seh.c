// RUN: %clang_cc1 -std=c23 -fdefer-ts -fms-compatibility -triple x86_64-windows-msvc -fsyntax-only -verify %s

void f() {
  __try {
    defer {
      __leave; // expected-error {{cannot __leave a defer statement}}
    }
  } __finally {}

  __try {
    defer {
      __try {
        __leave;
      } __finally {}
    }
  } __finally {}
}
