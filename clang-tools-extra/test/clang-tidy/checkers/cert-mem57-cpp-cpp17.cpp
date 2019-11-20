// RUN: %check_clang_tidy %s -std=c++14 cert-mem57-cpp %t
// RUN: clang-tidy --extra-arg='-std=c++17' -checks='-*,cert-mem57-cpp' --warnings-as-errors='*' %s
// RUN: clang-tidy --extra-arg='-std=c++2a' -checks='-*,cert-mem57-cpp' --warnings-as-errors='*' %s

struct alignas(128) Vector {
  char Elems[128];
};

void f() {
  auto *V1 = new Vector;        // CHECK-MESSAGES: warning: allocation function returns a pointer with alignment {{[0-9]+}} but the over-aligned type being allocated requires alignment 128 [cert-mem57-cpp]
  auto *V1_Arr = new Vector[2]; // CHECK-MESSAGES: warning: allocation function returns a pointer with alignment {{[0-9]+}} but the over-aligned type being allocated requires alignment 128 [cert-mem57-cpp]
}
