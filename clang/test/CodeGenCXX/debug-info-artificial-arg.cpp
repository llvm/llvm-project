// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s

template<class X> class B {
public:
  explicit B(X* p = 0);
};

class A
{
public:
  A(int value) : m_a_value(value) {};
  A(int value, A* client_A) : m_a_value (value), m_client_A (client_A) {}

  virtual ~A() {}

private:
  int m_a_value;
  B<A> m_client_A;
};

int main(int argc, char **argv) {
  A reallyA (500);
}

// CHECK: ![[CLASSTYPE:.*]] = !DICompositeType(tag: DW_TAG_class_type, name: "A",
// CHECK-SAME:                                 identifier: "_ZTS1A"
// CHECK: ![[ARTARG:.*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTS1A",
// CHECK-SAME:                            DIFlagArtificial
// CHECK: !DISubprogram(name: "A", scope: !"_ZTS1A"
// CHECK-SAME:          line: 12
// CHECK-SAME:          DIFlagPublic
// CHECK: !DISubroutineType(types: [[FUNCTYPE:![0-9]*]])
// CHECK: [[FUNCTYPE]] = !{null, ![[ARTARG]], !{{.*}}, !{{.*}}}
