// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-pedantic %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -Wformat-pedantic %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -Wformat %s -verify=okay
// okay-no-diagnostics

extern "C" int printf(const char *, ...);
#define LOG(...) printf(__VA_ARGS__)

namespace N {
  enum class E { One };
}

struct S {
  N::E Type;
};

using uint32_t = unsigned;
enum class FixedE : uint32_t { Two };

void a(N::E NEVal, S *SPtr, S &SRef) {
  printf("%d", N::E::One); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:25-[[@LINE-2]]:25}:")"

  printf("%hd", N::E::One); // expected-warning{{format specifies type 'short' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:")"

  printf("%hu", N::E::One); // expected-warning{{format specifies type 'unsigned short' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:")"

  LOG("%d", N::E::One); // expected-warning{{format specifies type 'int' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:13}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:22-[[@LINE-2]]:22}:")"

  LOG("%s", N::E::One); // expected-warning{{format specifies type 'char *' but the argument has type 'N::E'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:10}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:")"

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

  printf("%u", FixedE::Two); //expected-warning{{format specifies type 'unsigned int' but the argument has type 'FixedE'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<uint32_t>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:27-[[@LINE-2]]:27}:")"
}
