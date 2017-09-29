// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

bool a = true;
bool b = false;

namespace pr34273 {
  char c = "clang"[true];
  char d = true["clang"];
}

