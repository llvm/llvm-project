// RUN: %clang_cc1 -std=c++11 -debug-info-kind=limited -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -debug-info-kind=line-tables-only -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-GMLT %s
// RUN: %clang_cc1 -std=c++11 -debug-info-kind=standalone -S -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-NOLIMIT %s

namespace A {
#line 1 "foo.cpp"
namespace B {
extern int i;
int f1() { return 0; }
void f1(int) { }
struct foo;
struct bar { };
typedef bar baz;
extern int var_decl;
void func_decl(void);
extern int var_fwd;
void func_fwd(void);
}
}
namespace A {
using namespace B;
}

using namespace A;
namespace E = A;
int B::i = f1();
int func(bool b) {
  if (b) {
    using namespace A::B;
    return i;
  }
  using namespace A;
  using B::foo;
  using B::bar;
  using B::f1;
  using B::i;
  using B::baz;
  namespace X = A;
  namespace Y = X;
  using B::var_decl;
  using B::func_decl;
  using B::var_fwd;
  using B::func_fwd;
  return i + X::B::i + Y::B::i;
}

namespace A {
using B::i;
namespace B {
int var_fwd = i;
}
inline namespace I {
int var_i;
}
}
namespace {
int anonymous;
}
void B::func_fwd() {
  anonymous = 0;
}

namespace C {
  void c();
}
void C::c() {}

// This should work even if 'i' and 'func' were declarations & not definitions,
// but it doesn't yet.

// CHECK: [[I:![0-9]+]] = distinct !DIGlobalVariable(name: "i",{{.*}} scope: [[NS:![0-9]+]],
// CHECK: [[NS]] = !DINamespace(name: "B", scope: [[CTXT:![0-9]+]])
// CHECK: [[CTXT]] = !DINamespace(name: "A", scope: null)
// CHECK: [[FOOCPP:.*]] = !DIFile(filename: "foo.cpp"
// CHECK: [[VAR_FWD:![0-9]+]] = distinct !DIGlobalVariable(name: "var_fwd",{{.*}} scope: [[NS]],
// CHECK-SAME:                                             line: 44
// CHECK-SAME:                                             isDefinition: true
// CHECK: distinct !DIGlobalVariable(name: "var_i",{{.*}} scope: [[INLINE:![0-9]+]],
// CHECK: [[INLINE]] = !DINamespace(name: "I", scope: [[CTXT]], exportSymbols: true)
// CHECK: !DINamespace(scope: null)
// CHECK: [[CU:![0-9]+]] = distinct !DICompileUnit(
// CHECK-SAME:                            imports: [[MODULES:![0-9]*]]
// CHECK: [[MODULES]] = !{[[M1:![0-9]+]], [[M2:![0-9]+]], [[M3:![0-9]+]], [[M4:![0-9]+]], [[M5:![0-9]+]], [[M6:![0-9]+]], [[M7:![0-9]+]], [[M8:![0-9]+]], [[M9:![0-9]+]], [[M10:![0-9]+]], [[M11:![0-9]+]], [[M12:![0-9]+]], [[M13:![0-9]+]], [[M14:![0-9]+]], [[M15:![0-9]+]], [[M16:![0-9]+]], [[M17:![0-9]+]]
// CHECK: [[M1]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[CTXT]], entity: [[NS]], line: 15)

// CHECK: [[M2]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[CU]], entity: [[CTXT]],
// CHECK: [[M3]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "E", scope: [[CU]], entity: [[CTXT]], line: 19)
// CHECK: [[M4]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[LEX2:![0-9]+]], entity: [[NS]], line: 23)
// CHECK: [[LEX2]] = distinct !DILexicalBlock(scope: [[LEX1:![0-9]+]], file: [[FOOCPP]],
// CHECK: [[LEX1]] = distinct !DILexicalBlock(scope: [[FUNC:![0-9]+]], file: [[FOOCPP]],

// CHECK: [[FUNC:![0-9]+]] = distinct !DISubprogram(name: "func",{{.*}} isDefinition: true
// CHECK: [[M5]] = !DIImportedEntity(tag: DW_TAG_imported_module, scope: [[FUNC]], entity: [[CTXT:![0-9]+]],
// CHECK: [[M6]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[FOO:![0-9]+]], line: 27)
// CHECK: [[FOO]] = !DICompositeType(tag: DW_TAG_structure_type, name: "foo",
// CHECK-SAME:                               line: 5
// CHECK-SAME:                               DIFlagFwdDecl
// CHECK: [[M7]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[BAR:![0-9]+]]
// CHECK: [[BAR]] = !DICompositeType(tag: DW_TAG_structure_type, name: "bar",
// CHECK-SAME:                               line: 6
// CHECK-SAME:                               DIFlagFwdDecl

// CHECK: [[M8]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[F1:![0-9]+]]
// CHECK: [[F1:![0-9]+]] = distinct !DISubprogram(name: "f1",{{.*}} line: 4
// CHECK-SAME:                           isDefinition: true
// CHECK: [[M9]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[I]]
// CHECK: [[M10]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[BAZ:![0-9]+]]
// CHECK: [[BAZ]] = !DIDerivedType(tag: DW_TAG_typedef, name: "baz", scope: [[NS]], file: [[FOOCPP]],
// CHECK-SAME:                     baseType: [[BAR]]
// CHECK: [[M11]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "X", scope: [[FUNC]], entity: [[CTXT]]
// CHECK: [[M12]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "Y", scope: [[FUNC]], entity: [[M11]]
// CHECK: [[M13]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[VAR_DECL:![0-9]+]]
// CHECK: [[VAR_DECL]] = !DIGlobalVariable(name: "var_decl", linkageName: "{{[^"]*var_decl[^"]*}}", scope: [[NS]],{{.*}} line: 8,
// CHECK: [[M14]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[FUNC_DECL:![0-9]+]]
// CHECK: [[FUNC_DECL]] = !DISubprogram(name: "func_decl",
// CHECK-SAME:                          scope: [[NS]], file: [[FOOCPP]], line: 9
// CHECK: [[M15]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[VAR_FWD:![0-9]+]]
// CHECK: [[M16]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[FUNC]], entity: [[FUNC_FWD:![0-9]+]]
// CHECK: [[FUNC_FWD]] = distinct !DISubprogram(name: "func_fwd",{{.*}} line: 53,{{.*}} isDefinition: true
// CHECK: [[M17]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: [[CTXT]], entity: [[I]]
// CHECK: distinct !DISubprogram(name: "c",{{.*}}, scope: ![[C:[0-9]+]],{{.*}}, line: 60,{{.*}} isDefinition: true
// CHECK: ![[C]] = !DINamespace(name: "C",

// CHECK-GMLT: [[CU:![0-9]+]] = distinct !DICompileUnit(
// CHECK-GMLT-SAME:                            emissionKind: LineTablesOnly,
// CHECK-GMLT-NOT:                             imports:

// CHECK-NOLIMIT: !DICompositeType(tag: DW_TAG_structure_type, name: "bar",{{.*}} line: 6,
// CHECK-NOLIMIT-NOT:              DIFlagFwdDecl
// CHECK-NOLIMIT-SAME:             ){{$}}
