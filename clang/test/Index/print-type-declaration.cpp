
class Test{};

int main()
{
  auto a = Test();
  auto b = a;
}

// RUN: c-index-test -test-print-type-declaration -std=c++11 %s | FileCheck %s
// CHECK: VarDecl=a:6:8 (Definition) [typedeclaration=Test] [typekind=Record]
// CHECK: VarDecl=b:7:8 (Definition) [typedeclaration=Test] [typekind=Record]
