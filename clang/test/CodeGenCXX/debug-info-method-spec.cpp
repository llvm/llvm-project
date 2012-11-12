// RUN: %clang -fverbose-asm -g -S %s -o - | grep DW_AT_specification
// Radar 9254491
class A {
public:
  void doSomething(int i) { ++i; }
};

void foo(A *a) {
  a->doSomething(2);
}
