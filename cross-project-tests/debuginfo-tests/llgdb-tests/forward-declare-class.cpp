// RUN: %clangxx %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %test_debuginfo %s %t.o
// Radar 9168773
// XFAIL: !system-darwin && gdb-clang-incompatibility

// DEBUGGER: ptype A
// Work around a gdb bug where it believes that a class is a
// struct if there aren't any methods - even though it's tagged
// as a class.
// CHECK: {{struct|class}} A {
// CHECK:        int MyData;
// CHECK-NEXT: }
class A;
class B {
public:
  void foo(const A *p);
};

B iEntry;

class A {
public:
  int MyData;
};

A irp;

