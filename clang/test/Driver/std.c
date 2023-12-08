// RUN: %clang -w -std=c99 -trigraphs -std=gnu99 %s -E -o - | FileCheck -check-prefix=OVERRIDE %s
// OVERRIDE: ??(??)
// RUN: %clang -w -std=c99 -ftrigraphs -std=gnu99 %s -E -o - | FileCheck -check-prefix=FOVERRIDE %s
// FOVERRIDE: ??(??)
// RUN: %clang -w -ansi %s -E -o - | FileCheck -check-prefix=ANSI %s
// ANSI: []
// RUN: %clang -w -ansi %s -fno-trigraphs -E -o - | FileCheck -check-prefix=ANSI-OVERRIDE %s
// ANSI-OVERRIDE: ??(??)
// RUN: %clang -w -std=gnu99 -trigraphs %s -E -o - | FileCheck -check-prefix=EXPLICIT %s
// EXPLICIT: []
// RUN: %clang -w -std=gnu99 -ftrigraphs %s -E -o - | FileCheck -check-prefix=FEXPLICIT %s
// FEXPLICIT: []
// RUN: %clang -w -ftrigraphs -fno-trigraphs %s -E -o - | FileCheck -check-prefix=ONOFF %s
// ONOFF: ??(??)
// RUN: %clang -w -fno-trigraphs -trigraphs %s -E -o - | FileCheck -check-prefix=OFFFON %s
// OFFFON: []

??(??)
