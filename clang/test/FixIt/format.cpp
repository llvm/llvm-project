// RUN: %clang_cc1 -fsyntax-only -verify -Wformat %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -Wformat-pedantic %s 2>&1 | FileCheck %s
// expected-no-diagnostics

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
  printf("%d", N::E::One);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:25-[[@LINE-2]]:25}:")"

  printf("%hd", N::E::One);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:")"

  printf("%hu", N::E::One);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:14}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:")"

  LOG("%d", N::E::One);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:13}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:22-[[@LINE-2]]:22}:")"

  LOG("%s", N::E::One);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:8-[[@LINE-1]]:10}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:22-[[@LINE-3]]:22}:")"

  printf("%d", NEVal);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:21-[[@LINE-2]]:21}:")"

  LOG("%d", NEVal);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:13-[[@LINE-1]]:13}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:")"

  printf(
      "%d",
      SPtr->Type
  );
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:")"

  LOG(
      "%d",
      SPtr->Type
  );
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:17}:")"

  printf("%d",
      SRef.Type);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:")"

  LOG("%d",
      SRef.Type);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:"static_cast<int>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:")"

  printf("%u", FixedE::Two);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"static_cast<uint32_t>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:27-[[@LINE-2]]:27}:")"
}
