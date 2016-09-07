// RUN: clang-tidy -checks=-*,modernize-use-nullptr %s -- --driver-mode=cl /DTEST1 /DFOO=foo /DBAR=bar | FileCheck -implicit-check-not="{{warning|error}}:" %s
int *a = 0;
// CHECK: :[[@LINE-1]]:10: warning: use nullptr
#ifdef TEST1
int *b = 0;
// CHECK: :[[@LINE-1]]:10: warning: use nullptr
#endif
#define foo 1
#define bar 1
#if FOO
int *c = 0;
// CHECK: :[[@LINE-1]]:10: warning: use nullptr
#endif
#if BAR
int *d = 0;
// CHECK: :[[@LINE-1]]:10: warning: use nullptr
#endif
