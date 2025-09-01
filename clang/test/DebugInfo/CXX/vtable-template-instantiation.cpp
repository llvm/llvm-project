// For CTemplate we check in case of:
// - Implicitly instantiate whole class by up-casting:
//   * The vtable is generated with comdat
//   * Its '_vtable$' is generated
// - Implicitly instantiate member function only:
//  # when optimized:
//   * The vtable is vanished (no declaration)
//   * Its '_vtable$' is generated without associating to vtable
//  # when non-optimized:
//   * The vtable is generated with comdat only if non-optimized
//   * Its '_vtable$' is generated regardless optimized or not
// - Define explicitly instantiation:
//   * The vtable is generated with comdat
//   * Its '_vtable$' is generated
// - Declare explicitly instantiation as extern:
//   * The vtable is declared
//   * Its '_vtable$' is generated only if optimized

struct CBase {
  virtual void f() noexcept {}
};

template <typename T>
struct CTemplate: CBase {
  void f() noexcept override;
  virtual ~CTemplate() noexcept;
};
template <typename T>
void CTemplate<T>::f() noexcept {}
template <typename T>
CTemplate<T>::~CTemplate() noexcept {}

#ifdef EXPLICIT
template struct CTemplate<void>;
#endif
#ifdef EXTERN
extern template struct CTemplate<void>;
#endif

CTemplate<void> *get(CBase *) noexcept;

int main() {
  CTemplate<void> Template;
#ifdef NOCAST
  get(nullptr)->f();
#else
  get(&Template)->f();
#endif

  return 0;
}

// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 %s -o -             | FileCheck %s -check-prefixes IMPLICIT
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 %s -o -             | FileCheck %s -check-prefixes IMPLICIT
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 %s -o - -DNOCAST    | FileCheck %s -check-prefixes NOCAST,NOCAST-O0
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 %s -o - -DNOCAST    | FileCheck %s -check-prefixes NOCAST,NOCAST-O1
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 %s -o - -DEXPLICIT  | FileCheck %s -check-prefixes EXPLICIT
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 %s -o - -DEXPLICIT  | FileCheck %s -check-prefixes EXPLICIT
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 %s -o - -DEXTERN    | FileCheck %s -check-prefixes EXTERN,EXTERN-O0
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O1 %s -o - -DEXTERN    | FileCheck %s -check-prefixes EXTERN,EXTERN-O1

// IMPLICIT: $_ZTV9CTemplateIvE = comdat any
// IMPLICIT: @_ZTV9CTemplateIvE = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[VTABLE_VAR:![0-9]*]]
// IMPLICIT-DAG: [[VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV9CTemplateIvE"
// IMPLICIT-DAG: !DIGlobalVariableExpression(var: [[VTABLE]], expr: !DIExpression())
// IMPLICIT-DAG: [[TYPE:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CTemplate<void>"
// IMPLICIT-DAG: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[TYPE]], file: {{.*}}, baseType: [[PVOID:![0-9]+]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
// IMPLICIT-DAG: [[PVOID]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// NOCAST-O0: $_ZTV9CTemplateIvE = comdat any
// NOCAST-O0: @_ZTV9CTemplateIvE = linkonce_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[VTABLE_VAR:![0-9]*]]
// NOCAST-O1-NOT: $_ZTV9CTemplateIvE
// NOCAST-O1-NOT: @_ZTV9CTemplateIvE
// NOCAST-DAG: [[VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV9CTemplateIvE"
// NOCAST-DAG: !DIGlobalVariableExpression(var: [[VTABLE]], expr: !DIExpression())
// NOCAST-DAG: [[TYPE:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CTemplate<void>"
// NOCAST-DAG: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[TYPE]], file: {{.*}}, baseType: [[PVOID:![0-9]+]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
// NOCAST-DAG: [[PVOID]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// EXPLICIT: $_ZTV9CTemplateIvE = comdat any
// EXPLICIT: @_ZTV9CTemplateIvE = weak_odr {{.*}}unnamed_addr constant {{.*}}, comdat, align 8, !dbg [[VTABLE_VAR:![0-9]*]]
// EXPLICIT-DAG: [[VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV9CTemplateIvE"
// EXPLICIT-DAG: [[VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[VTABLE]], expr: !DIExpression())
// EXPLICIT-DAG: [[TYPE:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CTemplate<void>"
// EXPLICIT-DAG: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[TYPE]], file: {{.*}}, baseType: [[PVOID:![0-9]+]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
// EXPLICIT-DAG: [[PVOID]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

// EXTERN-NOT: $_ZTV9CTemplateIvE
// EXTERN-O0: @_ZTV9CTemplateIvE = external {{.*}}unnamed_addr constant {{.*}}, align 8{{$}}
// EXTERN-O1: @_ZTV9CTemplateIvE = external {{.*}}unnamed_addr constant {{.*}}, align 8, !dbg [[VTABLE_VAR:![0-9]*]]
// EXTERN-O0-NOT: linkageName: "_ZTV9CTemplateIvE"
// EXTERN-O1-DAG: [[VTABLE:![0-9]+]] = distinct !DIGlobalVariable(name: "_vtable$", linkageName: "_ZTV9CTemplateIvE"
// EXTERN-O1-DAG: [[VTABLE_VAR]] = !DIGlobalVariableExpression(var: [[VTABLE]], expr: !DIExpression())
// EXTERN-O1-DAG: [[TYPE:![0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "CTemplate<void>"
// EXTERN-O1-DAG: !DIDerivedType(tag: DW_TAG_variable, name: "_vtable$", scope: [[TYPE]], file: {{.*}}, baseType: [[PVOID:![0-9]+]], flags: DIFlagPrivate | DIFlagArtificial | DIFlagStaticMember)
// EXTERN-O1-DAG: [[PVOID]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
