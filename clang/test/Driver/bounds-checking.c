// RUN: %clang -target x86_64-apple-darwin10 -fbounds-checking -### -fsyntax-only %s 2> %t
// RUN: FileCheck < %t %s
// RUN: %clang -target x86_64-apple-darwin10 -fbounds-checking=3 -### -fsyntax-only %s 2> %t
// RUN: FileCheck -check-prefix=CHECK2 < %t %s

// CHECK: "-fbounds-checking=1"
// CHECK2: "-fbounds-checking=3"
