//RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -o - %s | FileCheck %s
//CHECK: DILocalVariable(
class Foo
{
 public:
  int x;
  int y;
  Foo (int i, int j) { x = i; y = j; }
};


Foo foo(10, 11);

int main() {
  int Foo::* pmi = &Foo::y;
  return foo.*pmi;
}
