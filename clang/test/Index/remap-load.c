// RUN: c-index-test -test-load-source all -remap-file="%s,%S/Inputs/remap-load-to.c" %s | FileCheck %s

// CHECK: remap-load.c:1:5: FunctionDecl=foo:1:5 (Definition) Extent=[1:1 - 3:2]
// CHECK: remap-load.c:1:13: ParmDecl=parm1:1:13 (Definition) Extent=[1:9 - 1:18]
// CHECK: remap-load.c:1:26: ParmDecl=parm2:1:26 (Definition) Extent=[1:20 - 1:31]
// CHECK: remap-load.c:2:10: UnexposedExpr= Extent=[2:10 - 2:23]
// CHECK: remap-load.c:2:10: BinaryOperator= Extent=[2:10 - 2:23]
// CHECK: remap-load.c:2:10: UnexposedExpr=parm1:1:13 Extent=[2:10 - 2:15]
// CHECK: remap-load.c:2:10: DeclRefExpr=parm1:1:13 Extent=[2:10 - 2:15]
// CHECK: remap-load.c:2:18: DeclRefExpr=parm2:1:26 Extent=[2:18 - 2:23]
