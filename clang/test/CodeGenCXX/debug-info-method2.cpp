// RUN: %clang_cc1 -x c++ -debug-info-kind=limited -S -emit-llvm < %s | FileCheck %s
// Preserve type qualifiers in -flimit-debug-info mode.

// CHECK:  DW_TAG_const_type
class A {
public:
  int bar(int arg) const;
};

int A::bar(int arg) const{
  return arg+2;
}

int main() {
  A a;
  int i = a.bar(2);
  return i;
}
