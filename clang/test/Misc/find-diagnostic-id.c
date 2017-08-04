// RUN: diagtool find-diagnostic-id warn_unused_variable | FileCheck %s
// RUN: not diagtool find-diagnostic-id warn_unused_vars 2>&1 | FileCheck --check-prefix=ERROR %s

// CHECK: {{^[0-9]+$}}
// ERROR: error: invalid diagnostic 'warn_unused_vars'
