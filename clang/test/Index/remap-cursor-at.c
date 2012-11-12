// RUN: c-index-test -cursor-at=%s:1:15 -cursor-at=%s:2:21 -remap-file="%s;%S/Inputs/remap-load-to.c" %s | FileCheck %s

// CHECK: ParmDecl=parm1:1:13 (Definition)
// CHECK: DeclRefExpr=parm2:1:26
