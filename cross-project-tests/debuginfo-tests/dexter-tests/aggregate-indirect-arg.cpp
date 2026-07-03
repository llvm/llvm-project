// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %clang++ -std=gnu++11 -O0 -g %s -o %t
// RUN: %dexter -w --use-script \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s
// Radar 8945514

class SVal {
public:
  ~SVal() {}
  const void* Data;
  unsigned Kind;
};

void bar(SVal &v) {}
class A {
public:
  void foo(SVal v) { bar(v); } // !dex_label foo
};

int main() {
  SVal v;
  v.Data = 0;
  v.Kind = 2142;
  A a;
  a.foo(v);
  return 0;
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label foo}:
  !value v:
    Kind: 2142
  !value "v.Data == 0": "true"
...
*/
