// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -Wformat %s 2>&1 | FileCheck %s

extern "C" int printf(const char *, ...);

namespace N {
  enum class E { One };
}

void a() {
  printf("%d", N::E::One); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:25-[[@LINE-2]]:25}:")"

  printf("%hd", N::E::One);
  // CHECK: "static_cast<short>("

  printf("%hu", N::E::One);
  // CHECK: "static_cast<unsigned short>("
}
