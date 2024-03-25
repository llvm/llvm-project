// RUN: %clang %s -E -DCLI_MACRO=1 2>&1 | FileCheck %s

#define CLI_MACRO
// CHECK: macro_redefined.S{{.+}}: warning: 'CLI_MACRO' macro redefined
// CHECK: <command line>{{.+}}: note: previous definition is here

#define REGULAR_MACRO
#define REGULAR_MACRO 1
// CHECK: macro_redefined.S{{.+}}: warning: 'REGULAR_MACRO' macro redefined
// CHECK: macro_redefined.S{{.+}}: note: previous definition is here
