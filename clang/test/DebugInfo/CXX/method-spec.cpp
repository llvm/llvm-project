// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang --target=%itanium_abi_triple -fverbose-asm -g -S %s -o - | grep DW_AT_specification
class A {
public:
  void doSomething(int i) { ++i; }
};

void foo(A *a) {
  a->doSomething(2);
}
