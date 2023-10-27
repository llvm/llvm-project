// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics


struct StringPiece {
  template <typename T,
           typename = decltype(T())>
             StringPiece(T str) {}
};

void f(StringPiece utf8) {}

struct S {
};

void G() {
  const auto s = S{};
  StringPiece U{s};
}

