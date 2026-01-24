// Test that -fms-anonymous-structs is a CC1-only option and is accepted by CC1 without error.

// RUN: %clang_cc1 -triple powerpc-ibm-aix -fms-anonymous-structs %s -fsyntax-only 2>&1 | \
// RUN:     FileCheck --check-prefix=CC1-OK %s --allow-empty
// CC1-OK-NOT: error: unknown argument

// Test that multiple occurrences are handled
// RUN: %clang_cc1 -triple powerpc-ibm-aix -fms-anonymous-structs -fms-anonymous-structs %s -fsyntax-only 2>&1 | \
// RUN:     FileCheck --check-prefix=CC1-OK %s --allow-empty

// Test with other MS-related options
// RUN: %clang_cc1 -triple powerpc-ibm-aix -fms-extensions -fms-anonymous-structs %s -fsyntax-only 2>&1 | \
// RUN:     FileCheck --check-prefix=CC1-OK %s --allow-empty

// Test rejection of the unsupported negative form.
// RUN: not %clang_cc1 -triple powerpc-ibm-aix -fno-ms-anonymous-structs %s -fsyntax-only 2>&1 | \
// RUN:     FileCheck %s
// CHECK: error: unknown argument: '-fno-ms-anonymous-structs'
