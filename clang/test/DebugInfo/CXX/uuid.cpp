// RUN: %clang_cc1 -emit-llvm -fms-extensions -triple=x86_64-pc-win32 -debug-info-kind=limited %s -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions -triple=x86_64-unknown-unknown -debug-info-kind=limited %s -o - -std=c++11 2>&1 | FileCheck %s --check-prefix=CHECK-ITANIUM

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid2<GUID{12345678-1234-1234-1234-1234567890ab}>"
// CHECK-SAME:             templateParams: [[TGI2ARGS:![0-9]*]]
// CHECK: [[TGI2ARGS]] = !{[[TGI2ARG1:![0-9]*]]}
// CHECK: [[TGI2ARG1]] = !DITemplateValueParameter(
// CHECK-SAME:                                     type: [[CONST_GUID_REF:![0-9]*]]
// CHECK-SAME:                                     value: ptr @_GUID_12345678_1234_1234_1234_1234567890ab
// CHECK: [[CONST_GUID_REF]] = !DIDerivedType(tag: DW_TAG_reference_type,
// CHECK-SAME:                                baseType: [[CONST_GUID:![0-9]*]]
// CHECK: [[CONST_GUID]] = !DIDerivedType(tag: DW_TAG_const_type
// CHECK-SAME:                            baseType: [[GUID:![0-9]*]]
// CHECK: [[GUID]] = !DICompositeType(tag: DW_TAG_structure_type, name: "_GUID"

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid<&GUID{12345678-1234-1234-1234-1234567890ab}>"
// CHECK-SAME:             templateParams: [[TGIARGS:![0-9]*]]
// CHECK: [[TGIARGS]] = !{[[TGIARG1:![0-9]*]]}
// CHECK: [[TGIARG1]] = !DITemplateValueParameter(
// CHECK-SAME:                                    type: [[CONST_GUID_PTR:![0-9]*]]
// CHECK-SAME:                                    value: ptr @_GUID_12345678_1234_1234_1234_1234567890ab
// CHECK: [[CONST_GUID_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-SAME:                                baseType: [[CONST_GUID:![0-9]*]]
// CHECK-SAME:                                size: 64

// CHECK-ITANIUM: !DICompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid2<GUID{12345678-1234-1234-1234-1234567890ab}>"
// CHECK-ITANIUM-SAME:             identifier: "_ZTS10tmpl_guid2IL_Z42_GUID_12345678_1234_1234_1234_1234567890abEE"
// CHECK-ITANIUM: !DICompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid<&GUID{12345678-1234-1234-1234-1234567890ab}>"
// CHECK-ITANIUM-SAME:             identifier: "_ZTS9tmpl_guidIXadL_Z42_GUID_12345678_1234_1234_1234_1234567890abEEE"

struct _GUID {
  __UINT32_TYPE__ a; __UINT16_TYPE__ b, c; __UINT8_TYPE__ d[8];
};
template <const _GUID *>
struct tmpl_guid {
};

struct __declspec(uuid("{12345678-1234-1234-1234-1234567890ab}")) uuid;
tmpl_guid<&__uuidof(uuid)> tgi;

template <const _GUID &>
struct tmpl_guid2 {};
tmpl_guid2<__uuidof(uuid)> tgi2;
