// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=OGCG

struct __attribute__((__abi_tag__("cxx11"))) S { int i;};

template <typename F> auto f(int c, F fn) { return fn(c); }
S s(int) { return {}; }

void g() {
  // The 'i' isn't required to reproduce, but it prevents an NYI in call-conv lowering.
  int i;
  auto r = [i] { return s(f(1, [i](auto x) { return x; })); }();
  (void)r;
}
int main() { g(); }

// CIR-LABEL: cir.func no_inline lambda internal private dso_local @_ZZ1gvENK3$_0clB5cxx11Ev
// CIR-NOT: define
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> 
// CIR: cir.call @_Z1fIZZ1gvENK3$_0clEvEUlT_E_EDaiS1_(%[[ONE]], %{{.*}})

// LLVM-LABEL: define internal i32 @"_ZZ1gvENK3$_0clB5cxx11Ev"(
// LLVM-NOT: define
// LLVM: call noundef i32 @"_Z1fIZZ1gvENK3$_0clEvEUlT_E_EDaiS1_"(i32 noundef 1, i32 %{{.*}})

// Testing to make sure we emit this in particular as a definition.
// CIR-LABEL: cir.func no_inline internal private dso_local @_Z1fIZZ1gvENK3$_0clEvEUlT_E_EDaiS1_
// CIR-NOT: cir.func
// CIR: cir.call @_ZZZ1gvENK3$_0clEvENKUlT_E_clIiEEDaS0_

// LLVM-LABEL: define internal noundef i32 @"_Z1fIZZ1gvENK3$_0clEvEUlT_E_EDaiS1_"
// LLVM-NOT: define
// LLVM: call noundef i32 @"_ZZZ1gvENK3$_0clEvENKUlT_E_clIiEEDaS0_"

// CIR-LABEL: cir.func no_inline lambda internal private dso_local @_ZZZ1gvENK3$_0clEvENKUlT_E_clIiEEDaS0_
// LLVM-LABEL: define internal noundef i32 @"_ZZZ1gvENK3$_0clEvENKUlT_E_clIiEEDaS0_"

// OGCG-LABEL: define internal i32 @"_ZZ1gvENK3$_0clB5cxx11Ev"(
// OGCG-NOT: define
// OGCG: call noundef i32 @"_Z1fIZZ1gvENK3$_0clEvEUlT_E_EDaiS1_"(i32 noundef 1, i32 %{{.*}})

// Testing to make sure we emit this in particular as a definition.
// OGCG-LABEL: define internal noundef i32 @"_Z1fIZZ1gvENK3$_0clEvEUlT_E_EDaiS1_"
// OGCG-NOT: define
// OGCG: call noundef i32 @"_ZZZ1gvENK3$_0clEvENKUlT_E_clIiEEDaS0_"

// OGCG-LABEL: define internal noundef i32 @"_ZZZ1gvENK3$_0clEvENKUlT_E_clIiEEDaS0_"
