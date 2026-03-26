// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

class Foo {
 public:
  void foo() {
    int aa = 2;
    auto f = [=] {
      int aaa = a + aa;
    };
    f();
  }

 private:
  int a = 1;
};

int main() {
  Foo f;
  f.foo();

  return 0;
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_class_type, name: "<lambda_1>", {{.*}}, elements: ![[ELEMENT_TAG:[0-9]+]]
// CHECK: ![[ELEMENT_TAG]] = !{![[FOO_THIS:[0-9]+]], ![[FOO_AA:[0-9]+]], ![[FOO_OPERATOR:[0-9]+]]}
// CHECK-NEXT: ![[FOO_THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "__this", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[#]], size: [[#]])
// CHECK-NEXT: ![[FOO_AA]] = !DIDerivedType(tag: DW_TAG_member, name: "aa", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[#]], size: [[#]], offset: [[#]])
// CHECK-NEXT: ![[FOO_OPERATOR]] = !DISubprogram(name: "operator()", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], scopeLine: [[#]], flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
