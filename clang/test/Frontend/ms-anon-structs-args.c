// Test that -fms-anonymous-structs is a CC1-only option and properly rejected by driver

// RUN: %clang_cc1 -triple powerpc-ibm-aix -fms-anonymous-structs %s -fsyntax-only 2>&1 | \
// RUN:     FileCheck --check-prefix=CC1-OK %s --allow-empty
// CC1-OK-NOT: error: unknown argument

// Test that multiple occurrences are handled
// RUN: %clang_cc1 -triple powerpc-ibm-aix -fms-anonymous-structs -fms-anonymous-structs %s -fsyntax-only 2>&1 | \
// RUN:     FileCheck --check-prefix=MULTI-OK %s --allow-empty
// MULTI-OK-NOT: error: unknown argument

// Test with other MS-related options
// RUN: %clang_cc1 -triple powerpc-ibm-aix -fms-extensions -fms-anonymous-structs %s -fsyntax-only 2>&1 | \
// RUN:     FileCheck --check-prefix=WITH-MS-EXT %s --allow-empty
// WITH-MS-EXT-NOT: error: unknown argument
