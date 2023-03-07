// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -I%S/Inputs -Wno-return-stack-address -fclangir -fclangir-lifetime-check="history=all;history_limit=1" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

// This can be diagnosed by clang with -Wreturn-stack-address
auto g() {
  int i = 12; // expected-note {{declared here but invalid after function end}}
  return [&] { // expected-warning {{returned lambda captures local variable}}
    i += 100;
    return i;
  };
}

// This cannot be diagnosed by -Wreturn-stack-address
auto g2() {
  int i = 12; // expected-note {{declared here but invalid after function end}}
  auto lam = [&] {
    i += 100;
    return i;
  };
  return lam; // expected-warning {{returned lambda captures local variable}}
}