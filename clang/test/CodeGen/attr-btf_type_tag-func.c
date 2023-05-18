// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -DDOUBLE_BRACKET_ATTRS=1 -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s

#if DOUBLE_BRACKET_ATTRS
#define __tag1 [[clang::btf_type_tag("tag1")]]
#define __tag2 [[clang::btf_type_tag("tag2")]]
#define __tag3 [[clang::btf_type_tag("tag3")]]
#define __tag4 [[clang::btf_type_tag("tag4")]]
#else
#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))
#define __tag4 __attribute__((btf_type_tag("tag4")))
#endif

int __tag1 * __tag2 *foo(int __tag1 * __tag2 *arg) { return arg; }

// CHECK: distinct !DISubprogram(name: "foo", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L9:[0-9]+]]
// CHECK: ![[L9]] = !DISubroutineType(types: ![[L10:[0-9]+]]
// CHECK: ![[L10]] = !{![[L11:[0-9]+]], ![[L11]]}
// CHECK: ![[L11]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L12:[0-9]+]], size: [[#]], annotations: ![[L16:[0-9]+]]
// CHECK: ![[L12]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L13:[0-9]+]], size: [[#]], annotations: ![[L14:[0-9]+]]
// CHECK: ![[L13]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed
// CHECK: ![[L14]] = !{![[L15:[0-9]+]]}
// CHECK: ![[L15]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L16]] = !{![[L17:[0-9]+]]}
// CHECK: ![[L17]] = !{!"btf_type_tag", !"tag2"}
// CHECK: !DILocalVariable(name: "arg", arg: 1, scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L11]])
