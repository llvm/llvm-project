// RUN: diagtool find-diagnostic-id warn_unused_variable > %t; FileCheck %s < %t
// RUN: cat %t | xargs diagtool find-diagnostic-id | FileCheck %s --check-prefix=INVERSE
// RUN: not diagtool find-diagnostic-id warn_unused_vars 2>&1 | FileCheck --check-prefix=ERROR %s

// CHECK: {{^[0-9]+$}}
// INVERSE: warn_unused_variable
// ERROR: error: invalid diagnostic 'warn_unused_vars'
