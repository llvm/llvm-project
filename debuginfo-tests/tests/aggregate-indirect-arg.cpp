// RUN: %clangxx %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %clangxx %target_itanium_abi_host_triple %t.o -o %t.out
// RUN: %test_debuginfo %s %t.out
// Radar 8945514
// DEBUGGER: break 22
// DEBUGGER: r
// DEBUGGER: p v
// CHECK: ${{[0-9]+}} =
// CHECK:  Data ={{.*}} 0x0{{(0*)}}
// CHECK:  Kind = 2142

class SVal {
public:
  ~SVal() {}
  const void* Data;
  unsigned Kind;
};

void bar(SVal &v) {}
class A {
public:
  void foo(SVal v) { bar(v); }
};

int main() {
  SVal v;
  v.Data = 0;
  v.Kind = 2142;
  A a;
  a.foo(v);
  return 0;
}
