// RUN: %clang_cc1 -E -fdollars-in-identifiers %s 2>&1 | FileCheck %s --check-prefix=CHECK-DOLLARS
// RUN: %clang_cc1 -E %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-DOLLARS
// GH128939

#define FOO$ 10
#define STR(x) #x
#define STR2(x) STR(x)
const char *p = STR2(FOO$);

// CHECK-NO-DOLLARS: const char *p = "$ 10$";
// CHECK-DOLLARS: const char *p = "10";

#define STR3 STR(
const char *q = STR3$10);

// CHECK-NO-DOLLARS: const char *q = "$10";
// CHECK-DOLLARS: const char *q = STR3$10);
