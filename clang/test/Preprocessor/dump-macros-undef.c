// RUN: %clang_cc1 -E -dD %s | FileCheck %s
// PR7818

// CHECK: # 1 "{{.+}}.c"
#define X 3
// CHECK: #define X 3
#undef X
// CHECK: #undef X
