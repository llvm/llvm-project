// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mxcoff-roptr %s 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=ROPTR,LINK
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -c -mxcoff-roptr %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=ROPTR
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mxcoff-roptr -mno-xcoff-roptr %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=NO_ROPTR

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mxcoff-roptr %s 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=ROPTR,LINK
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -S -mxcoff-roptr %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=ROPTR
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=NO_ROPTR
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mxcoff-roptr -flto %s 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=NO_DATA_SECTION_ERR,ROPTR,LINK,LTO_ROPTR
// RUN: touch %t.o
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mxcoff-roptr %t.o 2>&1 | \
// RUN:     FileCheck %s --check-prefix=LINK

// RUN: not %clang -### --target=powerpc64le-unknown-linux-gnu -mxcoff-roptr \
// RUN:     %s 2>&1 | FileCheck %s --check-prefix=TARGET_ROPTR_ERR
// RUN: not %clang -### --target=powerpc64le-unknown-linux-gnu -mno-xcoff-roptr \
// RUN:     %s 2>&1 | FileCheck %s --check-prefix=TARGET_NOROPTR_ERR
// RUN: touch %t.o
// RUN: not %clang -### --target=powerpc64-ibm-aix-xcoff -mxcoff-roptr -shared \
// RUN:     %t.o 2>&1 | FileCheck %s --check-prefix=SHARED_ERR
// RUN: not %clang -### --target=powerpc64le-unknown-linux-gnu -mxcoff-roptr -flto \
// RUN:     %t.o 2>&1 | FileCheck %s --check-prefix=TARGET_ROPTR_ERR
// RUN: not %clang -### --target=powerpc64-ibm-aix-xcoff -mxcoff-roptr -flto -fno-data-sections \
// RUN:     %t.o 2>&1 | FileCheck %s --check-prefix=DATA_SECTION_ERR
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mno-xcoff-roptr -flto -fno-data-sections \
// RUN:     %t.o 2>&1 | FileCheck %s --check-prefix=NO_DATA_SECTION_ERR
// RUN: not %clang -### --target=powerpc64le-unknown-linux-gnu -mno-xcoff-roptr -flto \
// RUN:     %t.o 2>&1 | FileCheck %s --check-prefix=TARGET_NOROPTR_ERR

// DATA_SECTION_ERR: error: -mxcoff-roptr is supported only with -fdata-sections
// NO_DATA_SECTION_ERR-NOT: error: -mxcoff-roptr is supported only with -fdata-sections
// TARGET_ROPTR_ERR: error: unsupported option '-mxcoff-roptr' for target 'powerpc64le-unknown-linux-gnu'
// TARGET_NOROPTR_ERR: error: unsupported option '-mno-xcoff-roptr' for target 'powerpc64le-unknown-linux-gnu'
// SHARED_ERR: error: -mxcoff-roptr is not supported with -shared

// ROPTR: "-mxcoff-roptr"
// LINK: "-bforceimprw"
// LTO_ROPTR: "-bplugin_opt:-mxcoff-roptr"
// NO_ROPTR-NOT: "-mxcoff-roptr"
// NO_ROPTR-NOT: "-bforceimprw"
