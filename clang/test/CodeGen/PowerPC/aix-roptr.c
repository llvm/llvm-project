// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple=powerpc-ibm-aix-xcoff -mxcoff-roptr -fdata-sections \
// RUN:     -S <%s | FileCheck %s --check-prefix=CHECK32
// RUN: %clang_cc1 -triple=powerpc64-ibm-aix-xcoff -mxcoff-roptr -fdata-sections \
// RUN:     -S <%s | FileCheck %s --check-prefix=CHECK64
// RUN: not %clang_cc1 -triple=powerpc-ibm-aix-xcoff -mxcoff-roptr \
// RUN:     -S <%s 2>&1 | FileCheck %s --check-prefix=DATA_SECTION_ERR
// RUN: not %clang_cc1 -triple=powerpc64-ibm-aix-xcoff -mxcoff-roptr \
// RUN:     -S <%s 2>&1 | FileCheck %s --check-prefix=DATA_SECTION_ERR
// RUN: not %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -mxcoff-roptr \
// RUN:     %s 2>&1 | FileCheck %s --check-prefix=TARGET_ROPTR_ERR

char c1 = 10;
char* const c1_ptr = &c1;
// CHECK32:         .csect c1_ptr[RO],2
// CHECK32-NEXT:	.globl	c1_ptr[RO]
// CHECK32-NEXT:	.align	2
// CHECK32-NEXT:	.vbyte	4, c1[RW]

// CHECK64:         .csect c1_ptr[RO],3
// CHECK64-NEXT:	.globl	c1_ptr[RO]
// CHECK64-NEXT:	.align	3
// CHECK64-NEXT:	.vbyte	8, c1[RW]

// DATA_SECTION_ERR: error: -mxcoff-roptr is supported only with -fdata-sections
// TARGET_ROPTR_ERR: error: unsupported option '-mxcoff-roptr' for target 'powerpc64le-unknown-linux-gnu'
