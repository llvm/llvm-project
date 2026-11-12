// RUN: %clang_cc1 -E -fms-compatibility %s | FileCheck --check-prefix=CHECK-MS-COMPAT %s
// RUN: %clang_cc1 -E %s | FileCheck --check-prefix=CHECK-NO-MS-COMPAT %s

#define FN(x) L#x
#define F L "aaa"
void *v1 = FN(aaa);
void *v2 = F;
// CHECK-MS-COMPAT: void *v1 = L"aaa";
// CHECK-MS-COMPAT: void *v2 = L "aaa";
// CHECK-NO-MS-COMPAT: void *v1 = L "aaa";
// CHECK-NO-MS-COMPAT: void *v2 = L "aaa";
