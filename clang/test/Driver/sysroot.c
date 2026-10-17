// Check that --sysroot= also applies to header search paths.
// RUN: %clang -target i386-unk-unk --sysroot=/FOO -### -E %s 2> %t1
// RUN: FileCheck --check-prefix=CHECK-SYSROOTEQ < %t1 %s
// CHECK-SYSROOTEQ: "-cc1"{{.*}} "-isysroot" "{{[^"]*}}/FOO"

// Apple Darwin uses -isysroot as the syslib root, too.
// RUN: touch %t2.o
// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -isysroot /FOO -### %t2.o 2> %t2
// RUN: FileCheck --check-prefix=CHECK-APPLE-ISYSROOT < %t2 %s
// CHECK-APPLE-ISYSROOT: "-arch" "i386"{{.*}} "-syslibroot" "{{[^"]*}}/FOO"

// Check that honor --sysroot= over -isysroot, for Apple Darwin.
// RUN: touch %t3.o
// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -isysroot /FOO --sysroot=/BAR -### %t3.o 2> %t3
// RUN: FileCheck --check-prefix=CHECK-APPLE-SYSROOT < %t3 %s
// CHECK-APPLE-SYSROOT: "-arch" "i386"{{.*}} "-syslibroot" "{{[^"]*}}/BAR"

// An empty --sysroot= does not override -isysroot.
// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -isysroot /FOO --sysroot= -### %t3.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-APPLE-ISYSROOT %s

// An empty --sysroot= alone produces no -syslibroot.
// RUN: %clang -target i386-apple-darwin10 \
// RUN:   --sysroot= -### %t3.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-APPLE-EMPTY-SYSROOT %s
// CHECK-APPLE-EMPTY-SYSROOT: "-arch" "i386"
// CHECK-APPLE-EMPTY-SYSROOT-NOT: "-syslibroot"
