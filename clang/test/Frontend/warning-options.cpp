// RUN: %clang_cc1 -Wmonkey -Wno-monkey -Wno-unused-command-line-arguments \
// RUN:        -Wno-unused-command-line-argument -Wmodule-build -Werror-vla -Rmodule-built %s 2>&1 | FileCheck %s
// CHECK: unknown warning option '-Wmonkey'
// CHECK: unknown warning option '-Wno-monkey'
// CHECK: unknown warning option '-Wno-unused-command-line-arguments'; did you mean '-Wno-unused-command-line-argument'?
// CHECK: unknown warning option '-Wmodule-build'; did you mean '-Wmodule-conflict'?
// CHECK-NEXT: unknown -Werror warning specifier: '-Werror-vla'
// CHECK: unknown remark option '-Rmodule-built'; did you mean '-Rmodule-build'?
