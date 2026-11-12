enum Color {
  Red,
  Green,
  Blue,

  Rouge = Red
};

void PR17970(void (*)(int), float);

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-decls.c:1:6: EnumDecl=Color:1:6 (Definition) Extent=[1:1 - 7:2]
// CHECK: load-decls.c:2:3: EnumConstantDecl=Red:2:3 (Definition) Extent=[2:3 - 2:6]
// CHECK: load-decls.c:3:3: EnumConstantDecl=Green:3:3 (Definition) Extent=[3:3 - 3:8]
// CHECK: load-decls.c:4:3: EnumConstantDecl=Blue:4:3 (Definition) Extent=[4:3 - 4:7]
// CHECK: load-decls.c:6:3: EnumConstantDecl=Rouge:6:3 (Definition) Extent=[6:3 - 6:14]
// CHECK: load-decls.c:6:11: DeclRefExpr=Red:2:3 Extent=[6:11 - 6:14]
//
// CHECK: load-decls.c:9:6: FunctionDecl=PR17970:9:6 Extent=[9:1 - 9:35]
// CHECK: load-decls.c:9:21: ParmDecl=:9:21 (Definition) Extent=[9:14 - 9:27]
// CHECK: load-decls.c:9:26: ParmDecl=:9:26 (Definition) Extent=[9:23 - 9:26]
// CHECK: load-decls.c:9:34: ParmDecl=:9:34 (Definition) Extent=[9:29 - 9:34]
