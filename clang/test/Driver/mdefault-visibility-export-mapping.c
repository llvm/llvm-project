// RUN: %clang -### -S -mdefault-visibility-export-mapping=explicit --target=powerpc-ibm-aix %s 2>&1 | FileCheck %s
// RUN: %clang -### -S -mdefault-visibility-export-mapping=explicit --target=powerpc-unknown-linux %s 2>&1 | \
// RUN:   FileCheck -check-prefix=ERROR %s

// CHECK: "-mdefault-visibility-export-mapping=explicit"

// ERROR: unsupported option '-mdefault-visibility-export-mapping=explicit' for target 'powerpc-unknown-linux'
