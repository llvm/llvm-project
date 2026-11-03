// Test is line- and column-sensitive; see below.

export module export_decl;

namespace plain_ns {
  int plain_helper(int v) { return v + 1; }
}

export namespace exported_ns {
  int exported_helper(int v) { return v * 2; }
  struct ExportedStruct { int y; };
}

export int exported_direct(int v);
export struct DirectStruct { int z; };

// RUN: c-index-test -test-load-source all -x c++-module -std=c++20 %s | FileCheck %s

// NOTE: The CHECK lines below were calibrated against the actual
// c-index-test output produced by this tree (24.0.0git trunk with the
// ModuleExportDecl patch applied). The `-x c++-module` must precede %s:
// placed after the input file, the driver ignores it ("no effect" warning)
// and the TU is not parsed in module purview.

// CHECK: export-decl.cpp:5:11: Namespace=plain_ns:5:11 (Definition) Extent=[5:1 - 7:2]
// CHECK: export-decl.cpp:6:7: FunctionDecl=plain_helper:6:7 (Definition) Extent=[6:3 - 6:44]
// CHECK: export-decl.cpp:6:24: ParmDecl=v:6:24 (Definition) Extent=[6:20 - 6:25]
// CHECK: export-decl.cpp:6:27: CompoundStmt= Extent=[6:27 - 6:44]
// CHECK: export-decl.cpp:6:29: ReturnStmt= Extent=[6:29 - 6:41]
// CHECK: export-decl.cpp:6:36: BinaryOperator=+ Extent=[6:36 - 6:41]
// CHECK: export-decl.cpp:6:36: UnexposedExpr=v:6:24 Extent=[6:36 - 6:37]
// CHECK: export-decl.cpp:6:36: DeclRefExpr=v:6:24 Extent=[6:36 - 6:37]
// CHECK: export-decl.cpp:6:40: IntegerLiteral= Extent=[6:40 - 6:41]
// CHECK: export-decl.cpp:9:1: ModuleExportDecl=:9:1 (Definition) Extent=[9:1 - 12:2]
// CHECK: export-decl.cpp:9:18: Namespace=exported_ns:9:18 (Definition) Extent=[9:8 - 12:2]
// CHECK: export-decl.cpp:10:7: FunctionDecl=exported_helper:10:7 (Definition) Extent=[10:3 - 10:47]
// CHECK: export-decl.cpp:10:27: ParmDecl=v:10:27 (Definition) Extent=[10:23 - 10:28]
// CHECK: export-decl.cpp:10:30: CompoundStmt= Extent=[10:30 - 10:47]
// CHECK: export-decl.cpp:10:32: ReturnStmt= Extent=[10:32 - 10:44]
// CHECK: export-decl.cpp:10:39: BinaryOperator=* Extent=[10:39 - 10:44]
// CHECK: export-decl.cpp:10:39: UnexposedExpr=v:10:27 Extent=[10:39 - 10:40]
// CHECK: export-decl.cpp:10:39: DeclRefExpr=v:10:27 Extent=[10:39 - 10:40]
// CHECK: export-decl.cpp:10:43: IntegerLiteral= Extent=[10:43 - 10:44]
// CHECK: export-decl.cpp:11:10: StructDecl=ExportedStruct:11:10 (Definition) Extent=[11:3 - 11:35]
// CHECK: export-decl.cpp:11:31: FieldDecl=y:11:31 (Definition) Extent=[11:27 - 11:32] [access=public]
// CHECK: export-decl.cpp:14:1: ModuleExportDecl=:14:1 (Definition) Extent=[14:1 - 14:34]
// CHECK: export-decl.cpp:14:12: FunctionDecl=exported_direct:14:12 Extent=[14:8 - 14:34]
// CHECK: export-decl.cpp:14:32: ParmDecl=v:14:32 (Definition) Extent=[14:28 - 14:33]
// CHECK: export-decl.cpp:15:1: ModuleExportDecl=:15:1 (Definition) Extent=[15:1 - 15:38]
// CHECK: export-decl.cpp:15:15: StructDecl=DirectStruct:15:15 (Definition) Extent=[15:8 - 15:38]
// CHECK: export-decl.cpp:15:34: FieldDecl=z:15:34 (Definition) Extent=[15:30 - 15:35] [access=public]
