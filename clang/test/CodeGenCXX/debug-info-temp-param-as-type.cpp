// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -fdebug-template-parameter-as-type -triple x86_64-apple-darwin %s -o - | FileCheck %s


template <typename T>
struct TClass {
  TClass();
  void foo();
  T val_;
  int val2_;
};

template <typename T>
void TClass<T>::foo() {
  T tVar = 1;
  T* pT = &tVar;
  tVar++;
}

template <typename T>
T bar(T tp) {
  return tp;
}

int main () {
  TClass<int> a;
  a.val2_ = 3;
  a.foo();
  auto A = bar(42);
  TClass<double> b;
  return 0;
}

// CHECK: [[INT:![0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TClass<int>"
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "val_",{{.*}}baseType: [[TPARAM:![0-9]+]]
// CHECK: [[TPARAM]] = !DIDerivedType(tag: DW_TAG_template_type_parameter, name: "T", {{.*}}baseType: [[INT]])
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "val2_",{{.*}}baseType: [[INT]]

// CHECK: !DILocalVariable(name: "A",{{.*}}type: [[TPARAM]])

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TClass<double>"
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "val_",{{.*}}baseType: [[TPARAM2:![0-9]+]]
// CHECK: [[TPARAM2]] = !DIDerivedType(tag: DW_TAG_template_type_parameter, name: "T", {{.*}}baseType: [[DOUBLE:![0-9]+]])
// CHECK: [[DOUBLE]] = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)

// CHECK: distinct !DISubprogram(name: "foo"
// CHECK: !DILocalVariable(name: "tVar",{{.*}}type: [[TPARAM]])
// CHECK: !DILocalVariable(name: "pT",{{.*}}type: [[TPTR:![0-9]+]]
// CHECK: [[TPTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[TPARAM]]

// CHECK: distinct !DISubprogram(name: "bar<int>"
// CHECK: !DILocalVariable(name: "tp",{{.*}}type: [[TPARAM]])
