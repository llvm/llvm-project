// UNSUPPORTED: target={{.*-zos.*}}
// RUN: %clang -w -std=c99 -trigraphs -std=gnu99 %s -E -o - | FileCheck -check-prefix=OVERRIDE %s
// OVERRIDE: ??(??)
// RUN: %clang -w -std=c99 -ftrigraphs -std=gnu99 %s -E -o - | FileCheck -check-prefix=FOVERRIDE %s
// FOVERRIDE: ??(??)

??(??)
