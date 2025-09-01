// The debug info of vtable is attached conditionally to whether
// - Member functions are inlined or not
// - Definition of destructor is visible or not

struct CInlined {
  virtual void f1() noexcept {}
  virtual void f2() noexcept {}
  virtual ~CInlined() noexcept;
};
#ifndef NO_DTOR_BODY
inline CInlined::~CInlined() noexcept {}
#endif

struct CNoInline {
  virtual void g1() noexcept;
  virtual void g2() noexcept;
  virtual ~CNoInline() noexcept;
};

void CNoInline::g1() noexcept {}
void CNoInline::g2() noexcept {}
#ifndef NO_DTOR_BODY
CNoInline::~CNoInline() noexcept {}
#endif

struct CNoFnDef {
  virtual void h1() noexcept;
  virtual void h2() noexcept;
  virtual ~CNoFnDef() noexcept;
};

#ifndef NO_DTOR_BODY
CNoFnDef::~CNoFnDef() noexcept {}
#endif

void use(void *, ...);

int main() {
  CInlined Inlined;
  CNoInline NoInline;
  CNoFnDef NoFnDef;
  use(&Inlined, &NoInline, &NoFnDef);

  return 0;
}

// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 %s -o - | FileCheck %s -check-prefixes CHECK-HAS-DTOR,CHECK-HAS-DTOR-O0
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 %s -o - | FileCheck %s -check-prefixes CHECK-HAS-DTOR,CHECK-HAS-DTOR-O1
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 -DNO_DTOR_BODY %s -o - | FileCheck %s -check-prefixes CHECK-NO-DTOR,CHECK-NO-DTOR-O0
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 -DNO_DTOR_BODY %s -o - | FileCheck %s -check-prefixes CHECK-NO-DTOR,CHECK-NO-DTOR-O1

// CHECK-HAS-DTOR: $_ZTV8CInlined = comdat any
// CHECK-HAS-DTOR-NOT: $_ZTV9CNoInline
// CHECK-HAS-DTOR-NOT: $_ZTV8CNoFnDef

// CHECK-HAS-DTOR-DAG: @_ZTV8CInlined = linkonce_odr {{.*}}constant {{{ \[[^]]*\] } { \[[^]]*\] \[[^]]*\] }}}, comdat, align 8, !dbg [[INLINED_VTABLE_VAR:![0-9]+]]
// CHECK-HAS-DTOR-DAG: @_ZTV9CNoInline = {{.*}}constant {{{ \[[^]]*\] } { \[[^]]*\] \[[^]]*\] }}}, align 8, !dbg [[NOINLINE_VTABLE_VAR:![0-9]+]]
// CHECK-HAS-DTOR-O0-DAG: @_ZTV8CNoFnDef = external {{.*}}constant {{{ \[[^]]*\] }}}, align 8{{$}}
// CHECK-HAS-DTOR-O1-DAG: @_ZTV8CNoFnDef = external {{.*}}constant {{{ \[[^]]*\] }}}, align 8, !dbg [[NOFNDEF_VTABLE_VAR:![0-9]+]]

// CHECK-HAS-DTOR: !llvm.dbg.cu

// CHECK-HAS-DTOR-DAG: [[INLINED_VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CInlined"
// CHECK-HAS-DTOR-DAG: [[INLINED_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[INLINED_VTABLE]], expr: !DIExpression())
// CHECK-HAS-DTOR-DAG: [[INLINED:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CInlined"
// CHECK-HAS-DTOR-DAG: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[INLINED]], file: {{.*}}, baseType: {{![0-9]+}}, flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK-HAS-DTOR-DAG: [[NOINLINE_VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV9CNoInline"
// CHECK-HAS-DTOR-DAG: [[NOINLINE_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[NOINLINE_VTABLE]], expr: !DIExpression())
// CHECK-HAS-DTOR-DAG: [[NOINLINE:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CNoInline"
// CHECK-HAS-DTOR-DAG: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[NOINLINE]], file: {{.*}}, baseType: {{![0-9]+}}, flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK-HAS-DTOR-O1-DAG: [[NOFNDEF_VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CNoFnDef"
// CHECK-HAS-DTOR-O1-DAG: [[NOFNDEF_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[NOFNDEF_VTABLE]], expr: !DIExpression())

// CHECK-NO-DTOR-NOT: $_ZTV8CInlined
// CHECK-NO-DTOR-NOT: $_ZTV9CNoInline
// CHECK-NO-DTOR-NOT: $_ZTV8CNoFnDef

// CHECK-NO-DTOR-DAG: @_ZTV8CInlined = external {{.*}}constant {{.*}}, align 8{{$}}
// CHECK-NO-DTOR-DAG: @_ZTV9CNoInline = {{.*}}constant {{{ \[[^]]*\] } { \[[^]]*\] \[[^]]*\] }}}, align 8, !dbg [[NOINLINE_VTABLE_VAR:![0-9]+]]
// CHECK-NO-DTOR-O0-DAG: @_ZTV8CNoFnDef = external {{.*}}constant {{{ \[[^]]*\] }}}, align 8{{$}}
// CHECK-NO-DTOR-O1-DAG: @_ZTV8CNoFnDef = external {{.*}}constant {{{ \[[^]]*\] }}}, align 8, !dbg [[NOFNDEF_VTABLE_VAR:![0-9]+]]

// CHECK-NO-DTOR: !llvm.dbg.cu

// CHECK-NO-DTOR-DAG: [[NOINLINE_VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV9CNoInline"
// CHECK-NO-DTOR-DAG: [[NOINLINE_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[NOINLINE_VTABLE]], expr: !DIExpression())
// CHECK-NO-DTOR-DAG: [[NOINLINE:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CNoInline"
// CHECK-NO-DTOR-DAG: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[NOINLINE]], file: {{.*}}, baseType: {{![0-9]+}}, flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)

// CHECK-NO-DTOR-O1-DAG: [[NOFNDEF_VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV8CNoFnDef"
// CHECK-NO-DTOR-O1-DAG: [[NOFNDEF_VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[NOFNDEF_VTABLE]], expr: !DIExpression())
