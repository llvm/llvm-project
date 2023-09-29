// RUN: %clang_cc1 -fsyntax-only -verify -Wformat %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -Wformat %s 2>&1 | FileCheck %s

extern "C" int printf(const char *, ...);
#define LOG(...) printf(__VA_ARGS__)

namespace N {
  enum class E { One };
}

struct S {
  N::E Type;
};

void a(N::E NEVal, S *SPtr, S &SRef) {
  printf("%d", N::E::One); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:25-[[@LINE-2]]:25}:")"

  printf("%hd", N::E::One); // expected-warning{{format specifies type 'short' but the argument has type 'N::E'}}
  // CHECK: "static_cast<short>("

  printf("%hu", N::E::One); // expected-warning{{format specifies type 'unsigned short' but the argument has type 'N::E'}}
  // CHECK: "static_cast<unsigned short>("

  LOG("%d", N::E::One); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:13}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:22-[[@LINE-2]]:22}:")"

  printf("%d", NEVal); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:21-[[@LINE-2]]:21}:")"

  LOG("%d", NEVal); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:13}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:")"

  printf(
      "%d",
      SPtr->Type // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  );
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:")"

  LOG( // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
      "%d",
      SPtr->Type
  );
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:")"

  printf("%d",
      SRef.Type); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:")"

  LOG("%d", // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
      SRef.Type);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:")"
}
