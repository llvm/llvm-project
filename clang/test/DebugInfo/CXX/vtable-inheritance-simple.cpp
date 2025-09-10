// Simple inheritance case:
// For CBase and CDerived we check:
// - Generation of their vtables (including attributes).
// - Generation of their '_vtable$' data members:
//   * Correct scope and attributes

namespace NSP {
  struct CBase {
    unsigned B = 1;
    virtual void zero() {}
    virtual int one() { return 1; }
    virtual int two() { return 2; }
    virtual int three() { return 3; }
  };
}

struct CDerived : NSP::CBase {
  unsigned D = 2;
  void zero() override {}
  int two() override { return 22; };
  int three() override { return 33; }
};

int main() {
  NSP::CBase Base;
  CDerived Derived;

  return 0;
}

// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -mrelocation-model pic -pic-is-pie -debug-info-kind=limited -dwarf-version=5 -disable-O0-optnone -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -mrelocation-model pic -pic-is-pie -debug-info-kind=limited -dwarf-version=5 -disable-O0-optnone -disable-llvm-passes %s -o - | FileCheck %s --check-prefix=COFF

// CHECK: $_ZTVN3NSP5CBaseE = comdat any
// CHECK: $_ZTV8CDerived = comdat any

// CHECK: @_ZTVN3NSP5CBaseE = linkonce_odr {{dso_local|hidden}} unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[BASE_VTABLE_VAR:![0-9]*]]
// CHECK: @_ZTV8CDerived = linkonce_odr {{dso_local|hidden}} unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[DERIVED_VTABLE_VAR:![0-9]*]]
// COFF: @_ZTVN3NSP5CBaseE = linkonce_odr {{dso_local|hidden}} unnamed_addr constant {{.*}}, comdat, align 8
// COFF-NOT: !dbg
// COFF-SAME: {{$}}
// COFF: @_ZTV8CDerived = linkonce_odr {{dso_local|hidden}} unnamed_addr constant {{.*}}, comdat, align 8
// COFF-NOT: !dbg
// COFF-SAME: {{$}}

// CHECK: [[BASE_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[BASE_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[BASE_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN3NSP5CBaseE"

// CHECK: [[DERIVED_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[DERIVED_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-NEXT: [[DERIVED_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CDerived"

// CHECK: [[TYPE:![0-9]*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[DERIVED:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK: [[DERIVED]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CDerived"

// CHECK: [[BASE:![0-9]*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBase"

// CHECK: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[BASE]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
