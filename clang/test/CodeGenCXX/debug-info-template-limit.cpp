// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple %itanium_abi_triple %s -o - | FileCheck %s

// Check that this pointer type is TC<int>
// CHECK: ![[LINE:[0-9]+]] = !DICompositeType(tag: DW_TAG_class_type, name: "TC<int>"{{.*}}, identifier: "_ZTS2TCIiE")
// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTS2TCIiE"

template<typename T>
class TC {
public:
  TC(const TC &) {}
  TC() {}
};

TC<int> tci;
