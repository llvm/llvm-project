// RUN: %clang_cc1 -x c++ -debug-info-kind=limited -triple bpf-linux-gnu -emit-llvm %s -o - | FileCheck %s

namespace foo {

template <typename T>
struct S {
  T x;
};

extern S<char> s;

int test(void) {
  return s.x;
}

}  // namespace foo

// CHECK: distinct !DIGlobalVariable(name: "s", scope: ![[NAMESPACE:[0-9]+]],{{.*}} type: ![[STRUCT_TYPE:[0-9]+]], isLocal: false, isDefinition: false)
// CHECK: ![[NAMESPACE]] = !DINamespace(name: "foo", scope: null)
// CHECK: ![[STRUCT_TYPE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<char>",{{.*}}size: 8, flags: DIFlagTypePassByValue, elements: ![[ELEMENT_TYPE:[0-9]+]], templateParams: ![[TEMPLATE_TYPE:[0-9]+]], identifier: "_ZTSN3foo1SIcEE")
// CHECK: ![[ELEMENT_TYPE]] = !{![[ELEMENT_TYPE:[0-9]+]]}
// CHECK: ![[ELEMENT_TYPE]] = !DIDerivedType(tag: DW_TAG_member, name: "x",{{.*}} baseType: ![[BASE_TYPE:[0-9]+]], size: 8)
// CHECK: ![[BASE_TYPE]] = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
// CHECK: ![[TEMPLATE_TYPE]] = !{![[TEMPLATE_TYPE:[0-9]+]]}
// CHECK: ![[TEMPLATE_TYPE]] = !DITemplateTypeParameter(name: "T", type: ![[BASE_TYPE]])
