// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -I%S/Inputs -Wno-return-stack-address -fclangir -fclangir-lifetime-check="history=all;history_limit=1" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

auto g() {
  int i = 12; // expected-note {{declared here but invalid after function end}}
  return [&] { // expected-warning {{returned lambda captures local variable}}
    i += 100;
    return i;
  };
}