// RUN: %clangxx -c -target %itanium_abi_triple -g %s -emit-llvm -S -o - | FileCheck %s
// RUN: %clangxx -c -target %ms_abi_triple -g %s -emit-llvm -S -o - | FileCheck %s

struct Foo {
  int A;
  Foo() : A(1){};
};

struct Bar {
  int B;
  Bar() : B(2){};
};

struct Baz {
  int C;
  Baz() : C(3){};
};

struct Qux {
  int d() { return 4; }
  Qux() {};
};

struct Quux {
  int E;
  Quux() : E(5){};
};

typedef int(Qux::*TD)();
typedef int(Qux::*TD1)();
int Val = reinterpret_cast<Baz *>(0)->C;
int main() {
  Bar *PB = new Bar;
  TD d = &Qux::d;
  (void)reinterpret_cast<TD1>(d);

  return reinterpret_cast<Foo *>(PB)->A + reinterpret_cast<Quux *>(0)->E;
}

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Foo",
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Bar",
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Baz",
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Qux",
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Quux",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "TD",
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "TD1",
