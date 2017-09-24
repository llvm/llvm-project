// RUN: %clang -target i386-apple-darwin9 -### -S -msoft-float %s 2>&1 | FileCheck -check-prefix=TEST1 %s
// TEST1: "-no-implicit-float"

// RUN: %clang -target i386-apple-darwin9 -### -S -msoft-float -mno-soft-float %s 2>&1 | FileCheck -check-prefix=TEST2 %s
// TEST2-NOT: "-no-implicit-float"

// RUN: %clang -target i386-apple-darwin9 -### -S -mno-soft-float %s -msoft-float 2>&1 | FileCheck -check-prefix=TEST3 %s
// TEST3: "-no-implicit-float"

// RUN: %clang -target i386-apple-darwin9 -### -S -mno-implicit-float %s 2>&1 | FileCheck -check-prefix=TEST4 %s
// TEST4: "-no-implicit-float"

// RUN: %clang -target i386-apple-darwin9 -### -S -mno-implicit-float -mimplicit-float %s 2>&1 | FileCheck -check-prefix=TEST4A %s
// TEST4A-NOT: "-no-implicit-float"

// RUN: %clang -target i386-apple-darwin9 -### -S -mkernel %s 2>&1 | FileCheck -check-prefix=TEST5 %s
// TEST5: "-no-implicit-float"

// RUN: %clang -target i386-apple-darwin9 -### -S -mkernel -mno-soft-float %s 2>&1 | FileCheck -check-prefix=TEST6 %s
// TEST6-NOT: "-no-implicit-float"

// RUN: %clang -target armv7-apple-darwin10 -### -S -mno-implicit-float %s 2>&1 | FileCheck -check-prefix=TEST7 %s
// TEST7: "-no-implicit-float"

// RUN: %clang -target armv7-apple-darwin10 -### -S -mno-implicit-float -mimplicit-float %s 2>&1 | FileCheck -check-prefix=TEST8 %s
// TEST8-NOT: "-no-implicit-float"
