// RUN: c-index-test -code-completion-at=%s:7:1 -load %llvmshlibdir/PrintFunctionNames%pluginext -add-plugin print-fns %s | FileCheck %s
// REQUIRES: plugins, examples
// CHECK: macro definition:{{.*}}
// CHECK-NOT: top-level-decl: "x"

void x();
