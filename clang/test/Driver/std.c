// RUN: %clang -std=c99 -trigraphs -std=gnu99 %s -E -o - | FileCheck -check-prefix=OVERRIDE %s
// OVERRIDE: ??(??)
// RUN: %clang -ansi %s -E -o - | FileCheck -check-prefix=ANSI %s
// ANSI: []
// RUN: %clang -std=gnu99 -trigraphs %s -E -o - | FileCheck -check-prefix=EXPLICIT %s
// EXPLICIT: []

??(??)
