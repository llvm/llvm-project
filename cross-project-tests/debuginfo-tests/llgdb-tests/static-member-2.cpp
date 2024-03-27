// RUN: %clangxx %target_itanium_abi_host_triple -O0 -g %s -o %t -c
// RUN: %clangxx %target_itanium_abi_host_triple %t -o %t.out
// RUN: %test_debuginfo %s %t.out
// XFAIL: gdb-clang-incompatibility

// DEBUGGER: delete breakpoints
// DEBUGGER: break static-member-2.cpp:36
// DEBUGGER: r
// DEBUGGER: ptype C
// CHECK:      {{struct|class}} C {
// CHECK:      static const int a
// CHECK-NEXT: static int b;
// CHECK-NEXT: static int c;
// CHECK:      int d;
// CHECK-NEXT: }
// DEBUGGER: p C::a
// CHECK:  4
// DEBUGGER: p C::c
// CHECK: 15

// PR14471, PR14734

class C {
public:
  const static int a = 4;
  static int b;
  static int c;
  int d;
};

int C::c = 15;
const int C::a;

int main() {
    C instance_C;
    return C::a;
}
