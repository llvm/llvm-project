// RUN: %clang_cc1 -std=c++11 -debug-info-kind=standalone -emit-llvm -triple x86_64-apple-darwin %s -o %t
// RUN: cat %t | FileCheck %s -check-prefix=CHECK0
// RUN: cat %t | FileCheck %s -check-prefix=CHECK1
// RUN: cat %t | FileCheck %s -check-prefix=CHECK2
//
// This test ensures that we associate a declaration with the
// definition of the constructor for OuterClass. The declaration is
// necessary so the backend can emit the DW_AT_specification attribute
// for the definition.
//
// rdar://problem/13116508

class Foo;
class OuterClass
{
  static class InnerClass {
  public:
    InnerClass(); // Here createContextChain() generates a limited type for OuterClass.
  } theInnerClass;
// CHECK0: ![[DECL:[0-9]+]] = !DISubprogram(name: "OuterClass"
// CHECK0-SAME: line: [[@LINE+2]]
// CHECK0-SAME: spFlags: 0
  OuterClass(const Foo *); // line 10
};
OuterClass::InnerClass OuterClass::theInnerClass; // This toplevel decl causes InnerClass to be generated.
// CHECK0: !DISubprogram(name: "OuterClass"
// CHECK0-SAME: line: [[@LINE+3]]
// CHECK0-SAME: DISPFlagDefinition
// CHECK0-SAME: declaration: ![[DECL]]
OuterClass::OuterClass(const Foo *meta) { } // line 13






class Foo1;
class OuterClass1
{
  static class InnerClass1 {
  public:
    InnerClass1();
  } theInnerClass1;
// CHECK1: ![[DECL:[0-9]+]] = !DISubprogram(name: "Bar"
// CHECK1-SAME: line: [[@LINE+2]]
// CHECK1-SAME: spFlags: 0
  void Bar(const Foo1 *);
};
OuterClass1::InnerClass1 OuterClass1::theInnerClass1;
// CHECK1: !DISubprogram(name: "Bar"
// CHECK1-SAME: line: [[@LINE+3]]
// CHECK1-SAME: DISPFlagDefinition
// CHECK1-SAME: declaration: ![[DECL]]
void OuterClass1::Bar(const Foo1 *meta) { }





class Foo2;
class OuterClass2
{
  static class InnerClass2 {
  public:
    InnerClass2();
  } theInnerClass2;
// CHECK2: ![[DECL:[0-9]+]] = !DISubprogram(name: "~OuterClass2"
// CHECK2-SAME: line: [[@LINE+2]]
// CHECK2-SAME: spFlags: 0
  ~OuterClass2(); // line 10
};
OuterClass2::InnerClass2 OuterClass2::theInnerClass2;
// CHECK4: !DISubprogram(name: "~OuterClass2"
// CHECK4-SAME: line: [[@LINE+3]]
// CHECK4-SAME: DISPFlagDefinition
// CHECK4-SAME: declaration: ![[DECL]]
OuterClass2::~OuterClass2() { }
