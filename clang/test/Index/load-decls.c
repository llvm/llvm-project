enum Color {
  Red,
  Green,
  Blue,

  Rouge = Red
};

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-decls.c:1:6: EnumDecl=Color:1:6 (Definition) Extent=[1:1 - 7:2]
// CHECK: load-decls.c:2:3: EnumConstantDecl=Red:2:3 (Definition) Extent=[2:3 - 2:6]
// CHECK: load-decls.c:3:3: EnumConstantDecl=Green:3:3 (Definition) Extent=[3:3 - 3:8]
// CHECK: load-decls.c:4:3: EnumConstantDecl=Blue:4:3 (Definition) Extent=[4:3 - 4:7]
// CHECK: load-decls.c:6:3: EnumConstantDecl=Rouge:6:3 (Definition) Extent=[6:3 - 6:14]
// CHECK: load-decls.c:6:11: DeclRefExpr=Red:2:3 Extent=[6:11 - 6:14]
