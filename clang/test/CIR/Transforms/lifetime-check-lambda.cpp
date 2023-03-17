// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -I%S/Inputs -Wno-return-stack-address -fclangir -fclangir-lifetime-check="history=all;history_limit=1" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

// Check also implements:
// EXP61-CPP. A lambda object must not outlive any of its reference captured objects

// This can be diagnosed by clang with -Wreturn-stack-address
auto g() {
  int i = 12; // expected-note {{declared here but invalid after enclosing function ends}}
  return [&] { // expected-warning {{returned lambda captures local variable}}
    i += 100;
    return i;
  };
}

// This cannot be diagnosed by -Wreturn-stack-address
auto g2() {
  int i = 12; // expected-note {{declared here but invalid after enclosing function ends}}
  auto lam = [&] {
    i += 100;
    return i;
  };
  return lam; // expected-warning {{returned lambda captures local variable}}
}

auto g3(int val) {
  auto outer = [val] {
    int i = val; // expected-note {{declared here but invalid after enclosing lambda ends}}
    auto inner = [&] {
      i += 30;
      return i;
    };
    return inner; // expected-warning {{returned lambda captures local variable}}
  };
  return outer();
}