// Make sure the -march is passed down to cc1as.
// RUN: %clang -target i386-unknown-freebsd -### -c -integrated-as %s \
// RUN: -march=geode 2>&1 | FileCheck -check-prefix=TARGET %s

// TARGET: "-cc1as"
// TARGET: "-target-cpu" "geode"
