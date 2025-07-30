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
