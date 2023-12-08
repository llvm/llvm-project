@import HasWarnings;

#ifdef WITH_ERRORS
@import HasErrors;
#endif

float float_val;
double *double_ptr = &float_val;

// RUN: rm -rf %t %t.diag %t.out
// RUN: %clang -fmodules -fmodules-cache-path=%t/ModuleCache -I %S/Inputs/ModuleDiags -fsyntax-only %s --serialize-diagnostics %t.diag > /dev/null 2>&1
// RUN: c-index-test -read-diagnostics %t.diag > %t.out 2>&1
// RUN: FileCheck --input-file=%t.out %s

// CHECK: has_warnings.h:3:8: warning: incompatible pointer types initializing 'float *'
// CHECK: serialized-diags.m:1:9: note: while building module 'HasWarnings' imported from
// CHECK: serialized-diags.m:8:9: warning: incompatible pointer types initializing 'double *'
// CHECK: Number of diagnostics: 2

// RUN: rm -rf %t %t.diag_errors %t.out_errors
// RUN: not %clang -fmodules -fmodules-cache-path=%t/ModuleCache -I %S/Inputs/ModuleDiags -fsyntax-only -DWITH_ERRORS %s --serialize-diagnostics %t.diag_errors > /dev/null 2>&1
// RUN: c-index-test -read-diagnostics %t.diag_errors > %t.out_errors 2>&1
// RUN: FileCheck -check-prefix=CHECK-WITH-ERRORS --input-file=%t.out_errors %s

// CHECK-WITH-ERRORS: has_warnings.h:3:8: warning: incompatible pointer types initializing 'float *'
// CHECK-WITH-ERRORS: serialized-diags.m:1:9: note: while building module 'HasWarnings'
// CHECK-WITH-ERRORS: has_errors.h:2:13: error: redefinition of 'foo'
// CHECK-WITH-ERRORS: serialized-diags.m:4:9: note: while building module 'HasErrors'
// CHECK-WITH-ERRORS: has_errors.h:1:13: note: previous definition is here
// CHECK-WITH-ERRORS: serialized-diags.m:4:9: fatal: could not build module 'HasErrors'
// CHECK-WITH-ERRORS: Number of diagnostics: 3

