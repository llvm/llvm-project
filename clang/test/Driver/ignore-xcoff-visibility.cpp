// RUN: %clang -### -target powerpc-unknown-aix  -mignore-xcoff-visibility -S %s 2> %t.log
// RUN: FileCheck -check-prefix=CHECK %s < %t.log
 CHECK: {{.*}}clang{{.*}}" "-cc1"
 CHECK: "-mignore-xcoff-visibility"

// RUN: not %clang -mignore-xcoff-visibility -target powerpc-unknown-linux  %s  2>&1 | \
// RUN: FileCheck -check-prefix=ERROR %s

 ERROR: unsupported option '-mignore-xcoff-visibility' for target 'powerpc-unknown-linux'
