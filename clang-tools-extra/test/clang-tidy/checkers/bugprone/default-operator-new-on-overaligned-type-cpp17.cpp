// RUN: %check_clang_tidy %s -std=c++14 bugprone-default-operator-new-on-overaligned-type %t
// RUN: clang-tidy -checks='-*,bugprone-default-operator-new-on-overaligned-type' --extra-arg=-Wno-unused-variable --warnings-as-errors='*' %s -- -std=c++17 -faligned-allocation
// RUN: clang-tidy -checks='-*,bugprone-default-operator-new-on-overaligned-type' --extra-arg=-Wno-unused-variable --warnings-as-errors='*' %s -- -std=c++17 -faligned-allocation

struct alignas(128) Vector {
  char Elems[128];
};

void f() {
  auto *V1 = new Vector;        // CHECK-MESSAGES: warning: allocation function returns a pointer with alignment {{[0-9]+}} but the over-aligned type being allocated requires alignment 128 [bugprone-default-operator-new-on-overaligned-type]
  auto *V1_Arr = new Vector[2]; // CHECK-MESSAGES: warning: allocation function returns a pointer with alignment {{[0-9]+}} but the over-aligned type being allocated requires alignment 128 [bugprone-default-operator-new-on-overaligned-type]
}
