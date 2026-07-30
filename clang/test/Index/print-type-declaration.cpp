
class Test{};

int main()
{
  auto a = Test();
  auto b = a;
}

enum RegularEnum {};

enum class ScopedEnum {};

// RUN: c-index-test -test-print-type-declaration -std=c++11 %s | FileCheck %s
// CHECK: VarDecl=a:6:8 (Definition) [typedeclaration=Test] [typekind=Record]
// CHECK: VarDecl=b:7:8 (Definition) [typedeclaration=Test] [typekind=Record]
// CHECK: EnumDecl=RegularEnum:10:6 (Definition) [typedeclaration=RegularEnum] [typekind=Enum]
// CHECK: EnumDecl=ScopedEnum:12:12 (Definition) (scoped) [typedeclaration=ScopedEnum] [typekind=Enum]

