// REQUIRES: target={{x86_64.*-linux.*}}

// Simple inheritance case:
// For CBase and CDerived we check:
// - Generation of their vtables (including attributes).
// - Generation of their '_vtable$' data members:
//   * Correct scope and attributes

#ifdef BASE_CODE
#define BASE_DEF
#endif

#ifdef DERIVED_CODE
#define BASE_DEF
#define DERIVED_DEF
#endif

#ifdef MAIN_CODE
#define BASE_DEF
#define DERIVED_DEF
#endif

#ifdef BASE_DEF
namespace NSP {
  struct CBase {
    unsigned B = 1;
    virtual void zero();
    virtual int one();
    virtual int two();
    virtual int three();
  };
}
extern void fooBase();
#endif

#ifdef BASE_CODE
void NSP::CBase::zero() {}
int NSP::CBase::one() { return 1; }
int NSP::CBase::two() { return 2; };
int NSP::CBase::three() { return 3; }

#ifdef SYMBOL_AT_FILE_SCOPE
static NSP::CBase Base;
#else
void fooBase() {
  NSP::CBase Base;
}
#endif
#endif

#ifdef DERIVED_DEF
struct CDerived : NSP::CBase {
  unsigned D = 2;
  void zero() override;
  int two() override;
  int three() override;
};
extern void fooDerived();
#endif

#ifdef DERIVED_CODE
void CDerived::zero() {}
int CDerived::two() { return 22; };
int CDerived::three() { return 33; }

#ifdef SYMBOL_AT_FILE_SCOPE
static CDerived Derived;
#else
void fooDerived() {
  CDerived Derived;
}
#endif
#endif

#ifdef MAIN_CODE
int main() {
#ifdef SYMBOL_AT_FILE_SCOPE
  NSP::CBase Base;
  CDerived Derived;
#else
  fooBase();
  fooDerived();
#endif

  return 0;
}
#endif

// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g %s -o %t.simple-base.bc    -DBASE_CODE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g %s -o %t.simple-derived.bc -DDERIVED_CODE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g %s -o %t.simple-main.bc    -DMAIN_CODE
// RUN: llvm-link %t.simple-base.bc %t.simple-derived.bc %t.simple-main.bc -S -o %t.simple-combined.ll
// RUN: FileCheck --input-file=%t.simple-combined.ll -check-prefix=CHECK-ONE %s

// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g -flto %s -o %t.simple-base.bc    -DBASE_CODE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g -flto %s -o %t.simple-derived.bc -DDERIVED_CODE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g -flto %s -o %t.simple-main.bc    -DMAIN_CODE
// RUN: llvm-link %t.simple-base.bc %t.simple-derived.bc %t.simple-main.bc -S -o %t.simple-combined.ll
// RUN: FileCheck --input-file=%t.simple-combined.ll -check-prefix=CHECK-ONE %s

// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g %s -o %t.simple-base.bc    -DBASE_CODE    -DSYMBOL_AT_FILE_SCOPE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g %s -o %t.simple-derived.bc -DDERIVED_CODE -DSYMBOL_AT_FILE_SCOPE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g %s -o %t.simple-main.bc    -DMAIN_CODE    -DSYMBOL_AT_FILE_SCOPE
// RUN: llvm-link %t.simple-base.bc %t.simple-derived.bc %t.simple-main.bc -S -o %t.simple-combined.ll
// RUN: FileCheck --input-file=%t.simple-combined.ll -check-prefix=CHECK-TWO %s

// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g -flto %s -o %t.simple-base.bc    -DBASE_CODE    -DSYMBOL_AT_FILE_SCOPE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g -flto %s -o %t.simple-derived.bc -DDERIVED_CODE -DSYMBOL_AT_FILE_SCOPE
// RUN: %clang --target=x86_64-linux -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes -emit-llvm -c -g -flto %s -o %t.simple-main.bc    -DMAIN_CODE    -DSYMBOL_AT_FILE_SCOPE
// RUN: llvm-link %t.simple-base.bc %t.simple-derived.bc %t.simple-main.bc -S -o %t.simple-combined.ll
// RUN: FileCheck --input-file=%t.simple-combined.ll -check-prefix=CHECK-TWO %s

// CHECK-ONE: ${{_ZN3NSP5CBaseC2Ev|_ZN8CDerivedC2Ev}} = comdat any
// CHECK-ONE: ${{_ZN3NSP5CBaseC2Ev|_ZN8CDerivedC2Ev}} = comdat any

// CHECK-ONE: @_ZTV8CDerived = {{dso_local|hidden}} unnamed_addr constant {{.*}}, align 8, !dbg [[DERIVED_VTABLE_VAR:![0-9]*]]
// CHECK-ONE: @_ZTVN3NSP5CBaseE = {{dso_local|hidden}} unnamed_addr constant {{.*}}, align 8, !dbg [[BASE_VTABLE_VAR:![0-9]*]]

// CHECK-ONE: [[DERIVED_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[DERIVED_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-ONE-NEXT: [[DERIVED_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CDerived"

// CHECK-ONE: [[TYPE:![0-9]*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// CHECK-ONE: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[DERIVED:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
// CHECK-ONE: [[DERIVED]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CDerived"

// CHECK-ONE: [[BASE_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[BASE_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-ONE-NEXT: [[BASE_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN3NSP5CBaseE"

// CHECK-ONE: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[BASE:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
// CHECK-ONE: [[BASE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBase"

// CHECK-TWO: ${{_ZN3NSP5CBaseC2Ev|_ZN8CDerivedC2Ev}} = comdat any
// CHECK-TWO: ${{_ZN3NSP5CBaseC2Ev|_ZN8CDerivedC2Ev}} = comdat any

// CHECK-TWO: @_ZTVN3NSP5CBaseE = {{dso_local|hidden}} unnamed_addr constant {{.*}}, align 8, !dbg [[BASE_VTABLE_VAR:![0-9]*]]
// CHECK-TWO: @_ZTV8CDerived = {{dso_local|hidden}} unnamed_addr constant {{.*}}, align 8, !dbg [[DERIVED_VTABLE_VAR:![0-9]*]]

// CHECK-TWO: [[BASE_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[BASE_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-TWO-NEXT: [[BASE_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTVN3NSP5CBaseE"

// CHECK-TWO: [[TYPE:![0-9]*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// CHECK-TWO: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[BASE:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
// CHECK-TWO: [[BASE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CBase"

// CHECK-TWO: [[DERIVED_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[DERIVED_VTABLE:![0-9]*]], expr: !DIExpression())
// CHECK-TWO-NEXT: [[DERIVED_VTABLE]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CDerived"

// CHECK-TWO: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[DERIVED:![0-9]*]], file: {{.*}}, baseType: [[TYPE]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK-TWO: [[DERIVED]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CDerived"
