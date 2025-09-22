// Diamond inheritance case:
// For CBase, CLeft, CRight and CDerived we check:
// - Generation of their vtables (including attributes).
// - Generation of their '_vtable$' data members:
//   * Correct scope and attributes

namespace NSP {
  struct CBase {
    int B = 0;
    virtual char fooBase() { return 'B'; }
  };
}

namespace NSP_1 {
  struct CLeft : NSP::CBase {
    int M1 = 1;
    char fooBase() override { return 'O'; };
    virtual int fooLeft() { return 1; }
  };
}

namespace NSP_2 {
  struct CRight : NSP::CBase {
    int M2 = 2;
    char fooBase() override { return 'T'; };
    virtual int fooRight() { return 2; }
  };
}

struct CDerived : NSP_1::CLeft, NSP_2::CRight {
  int D = 3;
  char fooBase() override { return 'D'; };
  int fooDerived() { return 3; };
};

int main() {
  NSP::CBase Base;
  NSP_1::CLeft Left;
  NSP_2::CRight Right;
  CDerived Derived;

  return 0;
}

// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 -disable-llvm-passes %s -o - | FileCheck %s

// CHECK: $_ZTVN3NSP5CBaseE = comdat any
// CHECK: $_ZTVN5NSP_15CLeftE = comdat any
// CHECK: $_ZTVN5NSP_26CRightE = comdat any
// CHECK: $_ZTV8CDerived = comdat any

// CHECK: @_ZTVN3NSP5CBaseE = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[BASE_VTABLE_VAR:![0-9]*]]
// CHECK: @_ZTVN5NSP_15CLeftE = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[LEFT_VTABLE_VAR:![0-9]*]]
// CHECK: @_ZTVN5NSP_26CRightE = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[RIGHT_VTABLE_VAR:![0-9]*]]
// CHECK: @_ZTV8CDerived = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[DERIVED_VTABLE_VAR:![0-9]*]]

// CHECK: [[BASE_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[BASE_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[BASE_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN3NSP5CBaseE"

// CHECK: [[LEFT_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[LEFT_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[LEFT_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN5NSP_15CLeftE"

// CHECK: [[TYPE:![0-9]*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[LEFT:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK: [[LEFT]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CLeft"

// CHECK: [[BASE:![0-9]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBase"

// CHECK: [[RIGHT_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[RIGHT_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[RIGHT_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN5NSP_26CRightE"

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[RIGHT:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK: [[RIGHT]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CRight"

// CHECK: [[DERIVED_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[DERIVED_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[DERIVED_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CDerived"

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[DERIVED:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK: [[DERIVED]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CDerived"

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[BASE]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
