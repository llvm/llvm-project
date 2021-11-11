// NO-PIE-NOT: "-pie"
// PIE: "-pie"

// RUN: %clang %s -### -o %t.o 2>&1 --target=arm-linux-androideabi \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=arm-linux-android \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=arm-linux-android24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// RUN: %clang %s -### -o %t.o 2>&1 --target=mipsel-linux-android \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=mipsel-linux-android24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// RUN: %clang %s -### -o %t.o 2>&1 --target=i686-linux-android \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=i686-linux-android24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// RUN: %clang %s -### -o %t.o 2>&1 --target=aarch64-linux-android \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=aarch64-linux-android24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// RUN: %clang %s -### -o %t.o 2>&1 --target=arm64-linux-android \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=arm64-linux-android24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-android \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-android24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// RUN: %clang %s -### -o %t.o 2>&1 --target=x86_64-linux-android \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 --target=x86_64-linux-android24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// Override toolchain default setting.
// RUN: %clang %s -### -o %t.o 2>&1 -pie --target=arm-linux-androideabi \
// RUN:   | FileCheck --check-prefix=PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 -no-pie -pie --target=arm-linux-androideabi24 \
// RUN:   | FileCheck --check-prefix=PIE %s

// RUN: %clang %s -### -o %t.o 2>&1 -no-pie --target=arm-linux-androideabi24 \
// RUN:   | FileCheck --check-prefix=NO-PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 -nopie --target=arm-linux-androideabi24 \
// RUN:   | FileCheck --check-prefix=NO-PIE %s
// RUN: %clang %s -### -o %t.o 2>&1 -pie -no-pie --target=arm-linux-androideabi24 \
// RUN:   | FileCheck --check-prefix=NO-PIE %s

// Static/shared/relocatable disable -pie

// RUN: %clang %s -### --target=aarch64-linux-android -static 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-STATIC
// CHECK-STATIC-NOT: "-pie"
// CHECK-STATIC: -static

// RUN: %clang %s -### --target=aarch64-linux-android -shared 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-SHARED
// CHECK-SHARED-NOT: "-pie"
// CHECK-SHARED: "-shared"

// RUN: %clang %s -### --target=aarch64-linux-android -r 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE-NOT: "-pie"
// CHECK-RELOCATABLE: "-r"
