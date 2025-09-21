// Multiple inheritance case:
// For CBaseOne, CBaseTwo and CDerived we check:
// - Generation of their vtables (including attributes).
// - Generation of their '_vtable$' data members:
//   * Correct scope and attributes

namespace NSP_1 {
  struct CBaseOne {
    int B1 = 1;
    virtual int one() { return 1; }
    virtual int two() { return 2; }
    virtual int three() { return 3; }
  };
}

namespace NSP_2 {
  struct CBaseTwo {
    int B2 = 1;
    virtual int four() { return 4; }
    virtual int five() { return 5; }
    virtual int six() { return 6; }
  };
}

struct CDerived : NSP_1::CBaseOne, NSP_2::CBaseTwo {
  int D = 1;
  int two() override { return 22; };
  int six() override { return 66; }
};

int main() {
  NSP_1::CBaseOne BaseOne;
  NSP_2::CBaseTwo BaseTwo;
  CDerived Derived;

  return 0;
}

// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 -disable-llvm-passes %s -o - | FileCheck %s

// CHECK: $_ZTVN5NSP_18CBaseOneE = comdat any
// CHECK: $_ZTVN5NSP_28CBaseTwoE = comdat any
// CHECK: $_ZTV8CDerived = comdat any

// CHECK: @_ZTVN5NSP_18CBaseOneE = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[BASE_ONE_VTABLE_VAR:![0-9]*]]
// CHECK: @_ZTVN5NSP_28CBaseTwoE = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[BASE_TWO_VTABLE_VAR:![0-9]*]]
// CHECK: @_ZTV8CDerived = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[DERIVED_VTABLE_VAR:![0-9]*]]

// CHECK: [[BASE_ONE_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[BASE_ONE_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[BASE_ONE_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN5NSP_18CBaseOneE"

// CHECK: [[BASE_TWO_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[BASE_TWO_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[BASE_TWO_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN5NSP_28CBaseTwoE"

// CHECK: [[TYPE:![0-9]*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[BASE_TWO:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// check: [[BASE_TWO]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBaseTwo"

// CHECK: [[DERIVED_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[DERIVED_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK: [[DERIVED_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CDerived"

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[DERIVED:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK: [[DERIVED]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CDerived"

// CHECK: [[BASE_ONE:![0-9]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBaseOne"

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[BASE_ONE]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
