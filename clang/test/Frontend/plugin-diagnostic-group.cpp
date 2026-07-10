// Tests that plugin diagnostics emitted in the plugin's own group
// ("<plugin>-plugin") can be controlled by the user like built-in diagnostics:
// a warning silenced with -Wno-<group> / the -Wplugin umbrella and promoted
// with -Werror=<group>; a remark that is off until -R<group>, independent of
// -W; an error that a group flag cannot silence; a group `#pragma clang
// diagnostic`; and a misspelled group reported as unknown.

// RUN: split-file %s %t

// A warning is on by default and prints its group.
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -warn-decls %t/simple.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN %s
// A warning is silenced by -Wno-<group> and by the -Wno-plugin umbrella.
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -warn-decls \
// RUN:   -Wno-print-fns-plugin %t/simple.cpp 2>&1 | FileCheck --check-prefix=SILENT %s
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -warn-decls \
// RUN:   -Wno-plugin %t/simple.cpp 2>&1 | FileCheck --check-prefix=SILENT %s
// A warning is silenced by the -Wno-user-defined-warnings root over every group.
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -warn-decls \
// RUN:   -Wno-user-defined-warnings %t/simple.cpp 2>&1 | FileCheck --check-prefix=SILENT %s
// A warning is promoted by -Werror=<group>.
// RUN: not %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -warn-decls \
// RUN:   -Werror=print-fns-plugin %t/simple.cpp 2>&1 | FileCheck --check-prefix=WERROR %s

// A remark is off by default, enabled by -R<group>, and unaffected by -W.
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -remark-decls %t/simple.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=SILENT %s
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -remark-decls \
// RUN:   -Rprint-fns-plugin -Wno-print-fns-plugin %t/simple.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=REMARK %s

// An error cannot be silenced by a group flag.
// RUN: not %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -error-decls \
// RUN:   -Wno-print-fns-plugin %t/simple.cpp 2>&1 | FileCheck --check-prefix=ERROR %s

// A `#pragma clang diagnostic` resolves the group and scopes the change.
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -plugin-arg-print-fns -warn-decls %t/pragma.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=PRAGMA %s

// A -Wno-<x>-plugin that no loaded plugin claims is a misspelled option.
// RUN: %clang_cc1 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -Wno-bogus-plugin %t/simple.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=UNKNOWN %s

// REQUIRES: plugins, examples

//--- simple.cpp
void f();

// WARN: warning: suspicious top-level declaration 'f' [-Wprint-fns-plugin]
// SILENT-NOT: top-level declaration 'f'
// WERROR: error: suspicious top-level declaration 'f'
// REMARK: remark: saw top-level declaration 'f' [-Rprint-fns-plugin]
// ERROR: error: forbidden top-level declaration 'f'
// UNKNOWN: unknown warning option

//--- pragma.cpp
void before();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wprint-fns-plugin"
void during();
#pragma clang diagnostic pop
void after();

// PRAGMA: warning: suspicious top-level declaration 'before'
// PRAGMA-NOT: declaration 'during'
// PRAGMA: warning: suspicious top-level declaration 'after'
