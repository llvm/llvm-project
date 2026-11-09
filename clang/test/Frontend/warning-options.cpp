// RUN: %clang_cc1 -Wmonkey -Wno-monkey -Wno-unused-command-line-arguments \
// RUN:        -Wno-unused-command-line-argument -Wmodule-build -Werror-vla -Rmodule-built %s 2>&1 | FileCheck %s
// CHECK: unknown warning option '-Wmonkey'
// CHECK: unknown warning option '-Wno-monkey'
// CHECK: unknown warning option '-Wno-unused-command-line-arguments'; did you mean '-Wno-unused-command-line-argument'?
// CHECK: unknown warning option '-Wmodule-build'; did you mean '-Wmodule-conflict'?
// CHECK-NEXT: unknown -Werror warning specifier: '-Werror-vla'
// CHECK: unknown remark option '-Rmodule-built'; did you mean '-Rmodule-build'?

// RUN: %clang_cc1 -Wmonkey -Wno-monkey -Wno-unused-command-line-arguments \
// RUN:        -Wno-unused-command-line-argument -Wmodule-build -Werror-vla -Rmodule-built -fdiagnostics-format sarif %s 2>&1
// FIXME: unlike TextDiagnosticPrinter, SARIFDiagnosticPrinter can't yet emit
// diagnostics in this limited mode without SourceManager, etc.
// At the moment all we can do is to ensure we don't crash with and without
// assertions.

// RUN: %clang_cc1 -Werror=monkey -Wno-error=monkey -Wfatal-errors=monkey \
// RUN:        -Wno-fatal-errors=monkey -Wno-error=unused-command-line-arguments \
// RUN:        %s 2>&1 | FileCheck --check-prefix=SPECIFIER %s
// SPECIFIER: unknown warning option '-Werror=monkey'
// SPECIFIER-NEXT: unknown warning option '-Wno-error=monkey'
// SPECIFIER-NEXT: unknown warning option '-Wfatal-errors=monkey'
// SPECIFIER-NEXT: unknown warning option '-Wno-fatal-errors=monkey'
// SPECIFIER-NEXT: unknown warning option '-Wno-error=unused-command-line-arguments'; did you mean '-Wno-error=unused-command-line-argument'?
