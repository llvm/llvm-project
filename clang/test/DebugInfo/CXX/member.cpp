// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang --target=%itanium_abi_triple -fverbose-asm -g -S %s -o - | grep DW_ACCESS_public
class A {
public:
  int x;
}; 
A a;
