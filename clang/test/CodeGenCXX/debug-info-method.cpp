// RUN: %clang -fverbose-asm -g -S %s -o - | grep DW_ACCESS_protected
class A {
protected:
  int foo();
}; 
A a;
